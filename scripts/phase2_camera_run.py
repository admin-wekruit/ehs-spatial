"""Bounded monocular or explicit sensor-depth RGB-D ORB-SLAM3 experiment.

prepare emits the runner and a reviewable vendor patch, never applies it.
run selects a chronological [start, stop) clip; invoke again with --load-run
for a separate-process revisit. Ground truth is read only after process exit.
Raw inputs and earlier runs are never modified.
"""

import argparse
from collections import Counter
import difflib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from reconstruct_tum_room import associate, digest, metric_ate, read_rows


REVISION = "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
CONTRACT = "phase2-camera-v4"
DEFAULT_VENDOR = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2/vendor")

CPP = r'''// Isolated Panoptes experiment. Requires the generated diagnostic patch.
#include "System.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

#ifndef PHASE2_CAMERA_DIAGNOSTICS
#error Apply the reviewed phase2-camera-diagnostics.patch before building this runner.
#endif
#ifndef PHASE2_FEATURE_MASK
#error Apply the reviewed phase2-camera-mask.patch before building this runner.
#endif

using namespace ORB_SLAM3;
using std::set;

static std::ofstream output(const std::string &name) {
    std::ofstream stream(name);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream << std::setprecision(17);
    return stream;
}

static void pose(std::ostream &out, const Sophus::SE3f &twc) {
    const Eigen::Matrix4f value = twc.matrix();
    if (!value.allFinite()) throw std::runtime_error("nonfinite pose");
    out << '[';
    for (int i=0;i<4;++i) {
        if (i) out << ',';
        out << '[';
        for (int j=0;j<4;++j) { if (j) out << ','; out << value(i,j); }
        out << ']';
    }
    out << ']';
}

static void ids(std::ostream &out, const set<unsigned long> &values) {
    out << '[';
    bool first=true;
    for (auto id : values) { if (!first) out << ','; out << id; first=false; }
    out << ']';
}

static set<unsigned long> edge_ids(const set<KeyFrame*> &edges) {
    set<unsigned long> result;
    for (auto kf : edges) if (kf) result.insert(kf->mnId);
    return result;
}

// Called before any input and after joined shutdown, so inventory is quiescent.
static void inventory(Atlas *atlas, const std::string &filename,
                      set<unsigned long> &keyframes, set<unsigned long> &points) {
    auto out=output(filename);
    out << "{\"maps\":[";
    bool first_map=true;
    for (auto map : atlas->GetAllMaps()) {
        if (!map || map->IsBad()) continue;
        if (!first_map) out << ',';
        first_map=false;
        out << "{\"map_id\":" << map->GetId() << ",\"keyframes\":[";
        bool first_kf=true;
        for (auto kf : map->GetAllKeyFrames()) {
            if (!kf || kf->isBad()) continue;
            keyframes.insert(kf->mnId);
            if (!first_kf) out << ',';
            first_kf=false;
            out << "{\"id\":" << kf->mnId << ",\"source_frame_id\":" << kf->mnFrameId
                << ",\"timestamp\":" << kf->mTimeStamp << ",\"c2w_native_atlas\":";
            pose(out,kf->GetPoseInverse());
            out << ",\"merge_edges\":"; ids(out,edge_ids(kf->GetMergeEdges()));
            out << ",\"loop_edges\":"; ids(out,edge_ids(kf->GetLoopEdges()));
            out << '}';
        }
        set<unsigned long> map_points;
        for (auto mp : map->GetAllMapPoints())
            if (mp && !mp->isBad()) map_points.insert(mp->mnId);
        points.insert(map_points.begin(),map_points.end());
        out << "],\"map_point_ids\":"; ids(out,map_points); out << '}';
    }
    out << "]}\n";
}

static bool has_cross_edge(Map *map, const set<unsigned long> &loaded) {
    if (!map || loaded.empty()) return false;
    // ponytail: linear scan of this small room's keyframes; index if maps grow.
    for (auto kf : map->GetAllKeyFrames()) {
        if (!kf || kf->isBad()) continue;
        for (auto edge : kf->GetMergeEdges())
            if (edge && !edge->isBad() && edge->GetMap()==map &&
                bool(loaded.count(kf->mnId)) != bool(loaded.count(edge->mnId))) return true;
    }
    return false;
}

static void sparse_map(Map *map, const std::string &units) {
    auto json=output("map.sparse.json");
    auto ply=output("map.sparse.ply");
    std::vector<MapPoint*> points;
    for (auto mp : map->GetAllMapPoints())
        if (mp && !mp->isBad() && mp->GetMap()==map) points.push_back(mp);
    ply << "ply\nformat ascii 1.0\ncomment " << units << " map_id " << map->GetId()
        << "\nelement vertex " << points.size()
        << "\nproperty float x\nproperty float y\nproperty float z\nproperty uint id\nend_header\n";
    json << "{\"map_id\":" << map->GetId()
         << ",\"coordinate_frame\":\"final_native_atlas\",\"scale\":\"" << units
         << "\",\"columns\":[\"id\",\"x\",\"y\",\"z\"],\"points\":[";
    bool first=true;
    for (auto mp : points) {
        const Eigen::Vector3f xyz=mp->GetWorldPos();
        if (!xyz.allFinite() || mp->mnId>4294967295UL) throw std::runtime_error("invalid sparse point");
        if (!first) json << ',';
        first=false;
        json << '[' << mp->mnId << ',' << xyz.x() << ',' << xyz.y() << ',' << xyz.z() << ']';
        ply << xyz.x() << ' ' << xyz.y() << ' ' << xyz.z() << ' ' << mp->mnId << '\n';
    }
    json << "]}\n";
}

static double source_period(const std::vector<double>& stamps, size_t index) {
    // Same first/interior/last-frame rule as official mono_tum.cc.
    if(index+1<stamps.size()) return stamps[index+1]-stamps[index];
    return index ? stamps[index]-stamps[index-1] : 0.0;
}

int main(int argc,char **argv) {
    if (argc==2 && std::string(argv[1])=="--contract") {
        std::cout << "phase2-camera-v4\n"; return 0;
    }
    if (argc==2 && std::string(argv[1])=="--self-check-pacing") {
        const std::vector<double> times={10.0,10.03,10.08};
        if(std::abs(source_period(times,0)-.03)>1e-12 ||
           std::abs(source_period(times,1)-.05)>1e-12 ||
           std::abs(source_period(times,2)-.05)>1e-12 ||
           source_period({10.0},0)!=0) return 1;
        std::cout << "official next-interval pacing check passed\n"; return 0;
    }
    if (argc!=6) {
        std::cerr << "Usage: phase2_camera_runner VOCAB SETTINGS INPUT_TSV monocular|rgbd none|allow-mask (cwd=output)\n";
        return 64;
    }
    try {
        const std::string mode=argv[4];
        if(mode!="monocular" && mode!="rgbd") throw std::runtime_error("unknown sensor");
        const bool rgbd=mode=="rgbd";
        const std::string mask_mode=argv[5];
        if(mask_mode!="none" && mask_mode!="allow-mask") throw std::runtime_error("unknown mask mode");
        const bool masked=mask_mode=="allow-mask";
        const std::string units=rgbd?"sensor_depth_meters":"uncalibrated_monocular";
        std::ifstream input(argv[3]);
        if (!input) throw std::runtime_error("cannot read input TSV");
        std::vector<std::string> input_lines;
        std::vector<double> source_times;
        std::string line;
        while(std::getline(input,line)) {
            std::istringstream row(line);
            std::string index,time;
            if(!std::getline(row,index,'\t') || !std::getline(row,time,'\t'))
                throw std::runtime_error("invalid input TSV");
            const double stamp=std::stod(time);
            if(!std::isfinite(stamp) || (!source_times.empty() && stamp<=source_times.back()))
                throw std::runtime_error("nonmonotonic timestamp");
            input_lines.push_back(line); source_times.push_back(stamp);
        }
        System slam(argv[1],argv[2],rgbd?System::RGBD:System::MONOCULAR,false);
        Atlas *atlas=slam.Phase2AtlasForDiagnostics();
        set<unsigned long> loaded_kfs,loaded_points;
        inventory(atlas,"atlas.loaded.json",loaded_kfs,loaded_points);
        auto frames=output("frames.online.jsonl");
        unsigned processed=0;
        for(size_t frame=0;frame<input_lines.size();++frame) {
            std::istringstream row(input_lines[frame]);
            std::string index_text,time_text,path,depth_time_text,depth_path,mask_path,extra;
            if (!std::getline(row,index_text,'\t') || !std::getline(row,time_text,'\t') ||
                !std::getline(row,path,'\t') || path.empty()) throw std::runtime_error("invalid input TSV");
            if(rgbd && (!std::getline(row,depth_time_text,'\t') || !std::getline(row,depth_path,'\t')))
                throw std::runtime_error("missing paired depth input");
            if(masked && (!std::getline(row,mask_path,'\t') || mask_path.empty()))
                throw std::runtime_error("missing feature allow mask");
            if(std::getline(row,extra,'\t')) throw std::runtime_error("extra input TSV column");
            const auto index=std::stoul(index_text);
            const double stamp=std::stod(time_text);
            const double period=source_period(source_times,frame);
            cv::Mat bgr=cv::imread(path,cv::IMREAD_COLOR);
            if (bgr.empty() || bgr.cols!=640 || bgr.rows!=480)
                throw std::runtime_error("expected original 640x480 RGB image: "+path);
            cv::Mat depth;
            double depth_stamp=0;
            if(rgbd) {
                depth_stamp=std::stod(depth_time_text);
                if(!std::isfinite(depth_stamp) || std::abs(depth_stamp-stamp)>=0.02)
                    throw std::runtime_error("depth timestamp outside strict 20ms pair tolerance");
                depth=cv::imread(depth_path,cv::IMREAD_UNCHANGED);
                if(depth.empty() || depth.size()!=bgr.size() || depth.type()!=CV_16UC1)
                    throw std::runtime_error("expected original registered uint16 sensor depth");
            }
            cv::Mat allow_mask;
            int allowed_pixels=bgr.rows*bgr.cols;
            if(masked) {
                allow_mask=cv::imread(mask_path,cv::IMREAD_UNCHANGED);
                if(allow_mask.empty() || allow_mask.type()!=CV_8UC1 || allow_mask.size()!=bgr.size() ||
                   cv::countNonZero((allow_mask!=0)&(allow_mask!=255)))
                    throw std::runtime_error("expected source-domain binary 0/255 feature allow mask");
                allowed_pixels=cv::countNonZero(allow_mask);
            }
            const auto started=std::chrono::steady_clock::now();
            // Native RGBD.DepthMapFactor converts raw uint16 / 5000 to meters.
            const Sophus::SE3f tcw=rgbd?slam.TrackRGBD(bgr,depth,stamp,{},path,allow_mask):slam.TrackMonocular(bgr,stamp,{},path,allow_mask);
            const double tracking_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            const auto keys=slam.Phase2ExtractedKeysForDiagnostics();
            if(masked) for(const auto &key : keys) {
                const int x=cvRound(key.pt.x), y=cvRound(key.pt.y);
                if(x<0 || x>=allow_mask.cols || y<0 || y>=allow_mask.rows || !allow_mask.at<unsigned char>(y,x))
                    throw std::runtime_error("extractor returned a keypoint in an excluded region");
            }
            const int state=slam.GetTrackingState();
            const bool valid=state==Tracking::OK;
            KeyFrame *reference=slam.Phase2ReferenceForDiagnostics();
            Map *reference_map=reference ? reference->GetMap() : nullptr;
            set<unsigned long> tracked,old_tracked;
            if (valid) for (auto mp : slam.GetTrackedMapPoints()) {
                if (!mp || mp->isBad() || mp->Observations()<1) continue;
                tracked.insert(mp->mnId);
                if (loaded_points.count(mp->mnId)) old_tracked.insert(mp->mnId);
            }
            const bool cross_edge=valid && has_cross_edge(reference_map,loaded_kfs);
            frames << "{\"source_index\":" << index << ",\"timestamp\":" << stamp
                << ",\"state\":" << state << ",\"valid\":" << (valid?"true":"false");
            if(rgbd) frames << ",\"depth_timestamp\":" << depth_stamp;
            frames << ",\"dynamic_mask_applied\":" << (masked?"true":"false")
                   << ",\"allow_mask_pixels\":" << allowed_pixels
                   << ",\"extracted_keypoints\":" << keys.size();
            if(masked) frames << ",\"excluded_region_keypoints\":0";
            frames << ",\"c2w_online_native\":";
            if (valid) pose(frames,tcw.inverse()); else frames << "null";
            frames << ",\"reference_keyframe_id\":";
            if (reference) frames << reference->mnId; else frames << "null";
            frames << ",\"reference_map_id_observed_after_tracking\":";
            if (reference_map) {
                std::unique_lock<std::mutex> lock(reference_map->mMutexMapUpdate);
                frames << reference_map->GetId();
            } else frames << "null";
            frames << ",\"tracked_map_points\":" << tracked.size()
                << ",\"loaded_map_points_tracked\":" << old_tracked.size()
                << ",\"cross_session_merge_edge_observed\":" << (cross_edge?"true":"false")
                << ",\"tracking_seconds\":"
                << tracking_seconds
                << ",\"source_interval_seconds\":" << period
                << "}\n";
            frames.flush();
            ++processed;
            if (processed%100==0) std::cout << "phase2 processed " << processed << std::endl;
            if (tracking_seconds<period) std::this_thread::sleep_for(std::chrono::duration<double>(period-tracking_seconds));
        }
        if (!processed) throw std::runtime_error("empty input");
        slam.Shutdown(); // Generated patch drains/joins producers and every GBA first.
        set<unsigned long> final_kfs,final_points;
        inventory(atlas,"atlas.final.json",final_kfs,final_points);
        Map *selected=nullptr;
        for (auto map : atlas->GetAllMaps())
            if (!map->IsBad() && map->KeyFramesInMap()>0 &&
                (!selected || map->KeyFramesInMap()>selected->KeyFramesInMap())) selected=map;
        if (!selected) { std::cerr << "No nonempty final map\n"; return 2; }
        sparse_map(selected,units); // Exactly one map, in the same coordinates as its final keyframes.
        auto keyframes=selected->GetAllKeyFrames();
        std::sort(keyframes.begin(),keyframes.end(),KeyFrame::lId);
        auto export_info=output("trajectory.export.json");
        export_info << "{\"map_id\":" << selected->GetId()
            << ",\"origin_keyframe_id\":" << keyframes.front()->mnId
            << ",\"atlas_from_euroc\":";
        pose(export_info,keyframes.front()->GetPoseInverse());
        export_info << ",\"timestamp_units\":\"nanoseconds\",\"position_units\":\"" << units << "\"}\n";
        slam.SaveTrajectoryEuRoC("trajectory.final.euroc.txt",selected);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "phase2 failed: " << error.what() << std::endl;
        return 1; // Preserve partial ledger; never serialize a live failed process.
    }
}
'''


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def mask_patch(orb):
    """Pass a source-pixel allow mask to the shared extractor; do not apply."""
    names = [f"{folder}/{name}.{suffix}" for folder, suffix in [("include", "h"), ("src", "cc")]
             for name in ["System", "Tracking", "Frame"]]
    originals = {name: (orb / name).read_text() for name in names}
    texts = originals.copy()

    def replace(name, old, new, count=1):
        if texts[name].count(old) != count:
            raise ValueError(f"Mask patch anchor mismatch: {name}: {old}")
        texts[name] = texts[name].replace(old, new)

    for name, methods in [("System", ["TrackRGBD", "TrackMonocular"]),
                          ("Tracking", ["GrabImageRGBD", "GrabImageMonocular"])]:
        for method in methods:
            for folder, suffix in [("include", "h"), ("src", "cc")]:
                path = f"{folder}/{name}.{suffix}"
                lines = [line for line in texts[path].splitlines() if
                         (f" {method}(" if suffix == "h" else f" {name}::{method}(") in line]
                if len(lines) != 1:
                    raise ValueError(f"Expected one declaration: {path} {method}")
                old = lines[0]
                new = old.replace('string filename="")', 'string filename="", const cv::Mat &allowMask=cv::Mat())') if suffix == "h" and name == "System" else old.replace(
                    'string filename)', 'string filename, const cv::Mat &allowMask' + ('=cv::Mat()' if suffix == "h" else '') + ')')
                replace(path, old, new)
    replace("include/System.h", "    int GetTrackingState();", """    int GetTrackingState();
#define PHASE2_FEATURE_MASK 1
    std::vector<cv::KeyPoint> Phase2ExtractedKeysForDiagnostics() const;""")
    replace("src/System.cc", "int System::GetTrackingState()", """std::vector<cv::KeyPoint> System::Phase2ExtractedKeysForDiagnostics() const {
    return mpTracker->mCurrentFrame.mvKeys;
}

int System::GetTrackingState()""")
    replace("src/System.cc", "    cv::Mat imToFeed = im.clone();", """    CV_Assert(allowMask.empty() || (allowMask.type()==CV_8UC1 && allowMask.size()==im.size()));
    if(!allowMask.empty() && settings_ && settings_->needToResize())
        CV_Error(cv::Error::StsBadArg, "Feature masks require original camera pixels without resizing");
    cv::Mat imToFeed = im.clone();""", 2)
    replace("src/System.cc", "GrabImageRGBD(imToFeed,imDepthToFeed,timestamp,filename)",
            "GrabImageRGBD(imToFeed,imDepthToFeed,timestamp,filename,allowMask)")
    replace("src/System.cc", "GrabImageMonocular(imToFeed,timestamp,filename)",
            "GrabImageMonocular(imToFeed,timestamp,filename,allowMask)")
    for path in ["include/Frame.h", "src/Frame.cc"]:
        declarations = [line for line in texts[path].splitlines() if
                        ("Frame(const cv::Mat &imGray," if path.endswith(".h") else "Frame::Frame(const cv::Mat &imGray,") in line]
        if len(declarations) != 2:
            raise ValueError("Expected mono and RGB-D Frame constructors")
        for old in declarations:
            new = old.replace("const IMU::Calib &ImuCalib = IMU::Calib())", "const IMU::Calib &ImuCalib = IMU::Calib(), const cv::Mat &allowMask=cv::Mat())") if path.endswith(".h") else old.replace(
                "const IMU::Calib &ImuCalib)", "const IMU::Calib &ImuCalib, const cv::Mat &allowMask)")
            replace(path, old, new)
    for old in [line for line in texts["src/Tracking.cc"].splitlines() if
                "Frame(mImGray,imDepth," in line or "Frame(mImGray,timestamp," in line]:
        if "&mLastFrame,*mpImuCalib" in old:
            new = old.replace("&mLastFrame,*mpImuCalib);", "&mLastFrame,*mpImuCalib,allowMask);")
        else:
            new = old.replace(");", ",nullptr,IMU::Calib(),allowMask);")
        replace("src/Tracking.cc", old, new)
    replace("include/Frame.h", "void ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1);",
            "void ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1, const cv::Mat &allowMask=cv::Mat());")
    replace("src/Frame.cc", "void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)",
            "void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1, const cv::Mat &allowMask)")
    replace("src/Frame.cc", "ExtractORB(0,imGray,0,0);", "ExtractORB(0,imGray,0,0,allowMask);")
    replace("src/Frame.cc", "ExtractORB(0,imGray,0,1000);", "ExtractORB(0,imGray,0,1000,allowMask);")
    for old in [line for line in texts["src/Frame.cc"].splitlines() if "(&Frame::ExtractORB," in line]:
        replace("src/Frame.cc", old, old.replace(");", ",cv::Mat());"))
    replace("src/Frame.cc", "(im,cv::Mat(),mvKeys", "(im,allowMask,mvKeys", 2)
    return "".join(line for name in names for line in difflib.unified_diff(
        originals[name].splitlines(True), texts[name].splitlines(True), fromfile="a/"+name, tofile="b/"+name))


def prepare_allow_masks(raw_path, video_manifest_path, dataset, rows, selected, output):
    """Validate completed native observations and bind their union to original RGB."""
    import cv2
    from build_video_pose_preview import decode_coco_rle, source_spans

    raw_path, video_manifest_path = raw_path.resolve(), video_manifest_path.resolve()
    raw = json.loads(raw_path.read_text())
    status = json.loads((raw_path.parent / "run.json").read_text())
    parent_path = raw_path.parent / "source-manifest.json"
    parent = json.loads(parent_path.read_text())
    video = json.loads(video_manifest_path.read_text())
    if status.get("status") != "execution_complete" or status.get("raw_sha256") != digest(raw_path):
        raise ValueError("Masks require completed, unchanged native observations")
    if parent["sourceStartFrame"] != 0 or len(raw["frames"]) != len(rows):
        raise ValueError("Feature masks must cover every original RGB frame")
    spans = source_spans(raw, parent)
    expected_mapping = "MP4 frame i corresponds to data row i of original rgb.txt (zero-based, excluding comment lines). Every RGB input frame appears once and in order."
    playback = video["playback"]
    if ((video_manifest_path.parent / video["original_root"]).resolve() != dataset
            or playback["sha256"] != parent["sourceSha256"]
            or playback["frame_mapping"] != expected_mapping
            or playback["input_frames"] != len(rows)
            or playback["decoded_frames"] != len(rows)
            or video["streams"]["rgb"]["count"] != len(rows)
            or (raw["input"]["height"], raw["input"]["width"]) != (480, 640)):
        raise ValueError("Mask video is not the verified original RGB pixel/frame domain")
    originals = {f["path"]: f["sha256"] for f in video["original_files"]}
    if originals.get("rgb.txt") != digest(dataset / "rgb.txt"):
        raise ValueError("Video and camera RGB index differ")
    for _, relative in rows:
        if originals.get(relative) != digest(dataset / relative):
            raise ValueError("Video and camera RGB source differ: " + relative)
    masks = output / "feature-allow-masks"
    masks.mkdir()
    for frame in selected:
        index = frame["source_index"]
        observed = raw["frames"][index]
        if observed["source_frame_index"] != index:
            raise ValueError("Missing or reordered source mask")
        excluded = np.zeros((480, 640), bool)
        total = 0
        for obj in observed["objects"]:
            if obj["label"] != "person":
                raise ValueError("Expected native person segmentation")
            mask = decode_coco_rle(obj["rle"], height=480, width=640).astype(bool)
            if mask.shape != excluded.shape:
                raise ValueError("Mask dimensions differ from original RGB")
            excluded |= mask
            total += int(mask.sum())
        allow = np.where(excluded, 0, 255).astype(np.uint8)
        path = masks / f"{index:06d}.png"
        if not cv2.imwrite(str(path), allow) or not np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), allow):
            raise ValueError("Feature allow mask PNG changed during serialization")
        frame.update(allow_mask_path=str(path), allow_mask_sha256=digest(path),
                     allow_mask_wh=[640, 480], allow_mask_pixels=int(np.count_nonzero(allow)),
                     excluded_pixels=int(excluded.sum()), overlapping_instance_pixels=total-int(excluded.sum()),
                     mask_media_pts_seconds=spans[index][0])
    return {"observations_sha256": digest(raw_path), "source_manifest_sha256": digest(parent_path),
            "video_manifest_sha256": digest(video_manifest_path), "video_sha256": playback["sha256"],
            "operation": "Union of native person masks; allow=255, exclude=0. Overlapping IDs count once.",
            "pixel_transform": "identity; reencoded source RGB masks applied at original RGB pixel coordinates",
            "identity_required": False, "segmentation_quality": "requires_visual_review",
            "mask_support_policy": "ORBextractor rejects FAST candidates whose descriptor support intersects excluded pixels"}


def vendor_patch(orb):
    """Return an unapplied diff against the pinned source; no vendor mutations."""
    replacements = {
        "include/System.h": [("    int GetTrackingState();", """    int GetTrackingState();
#define PHASE2_CAMERA_DIAGNOSTICS 1
    // Experiment-only read access; inventory only before input or after shutdown.
    Atlas* Phase2AtlasForDiagnostics() const { return mpAtlas; }
    KeyFrame* Phase2ReferenceForDiagnostics() const;""")],
        "src/System.cc": [("    mpLocalMapper->RequestFinish();\n    mpLoopCloser->RequestFinish();", """    // Stop the producer before the consumer; both loops drain their queues.
    mpLocalMapper->RequestFinish();
    mptLocalMapping->join();
    mpLoopCloser->RequestFinish();
    mptLoopClosing->join();
    mpLoopCloser->Phase2JoinGBA();"""), ("int System::GetTrackingState()", """KeyFrame* System::Phase2ReferenceForDiagnostics() const {
    return mpTracker->mCurrentFrame.mpReferenceKF;
}

int System::GetTrackingState()""")],
        "src/LocalMapping.cc": [("        if(CheckFinish())\n            break;", "        if(CheckFinish() && !CheckNewKeyFrames())\n            break;")],
        "include/LoopClosing.h": [("    void RequestFinish();", """    void RequestFinish();
    // Called after the loop-closing thread has joined; no concurrent publishers.
    void Phase2JoinGBA() {
        if(mpThreadGBA) { mPhase2RetiredGBA.push_back(mpThreadGBA); mpThreadGBA=nullptr; }
        for(auto worker : mPhase2RetiredGBA) {
            if(worker->joinable()) worker->join();
            delete worker;
        }
        mPhase2RetiredGBA.clear();
    }"""), ("    std::thread* mpThreadGBA;", "    std::thread* mpThreadGBA;\n    std::vector<std::thread*> mPhase2RetiredGBA;")],
        "src/LoopClosing.cc": [("        if(CheckFinish()){", "        if(CheckFinish() && !CheckNewKeyFrames()){"),
            ("            mpThreadGBA->detach();\n            delete mpThreadGBA;", """            // Retain canceled workers so saving cannot race a detached optimizer.
            mPhase2RetiredGBA.push_back(mpThreadGBA);
            mpThreadGBA = nullptr;""")],
    }
    result=[]
    for relative, changes in replacements.items():
        original=subprocess.check_output(["git", "show", f"{REVISION}:{relative}"], cwd=orb, text=True)
        modified=original
        for old,new in changes:
            expected=3 if "mpThreadGBA->detach" in old else 1
            if modified.count(old)!=expected:
                raise ValueError(f"Pinned patch anchor mismatch: {relative}: {old}")
            modified=modified.replace(old,new)
        result.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                          fromfile="a/"+relative, tofile="b/"+relative))
    return "".join(result)


def prepare(args):
    args.runner_dir.mkdir(parents=True, exist_ok=True)
    (args.runner_dir / "phase2_camera_runner.cc").write_text(CPP)
    patch=args.runner_dir / "phase2-camera-diagnostics.patch"
    patch.write_text(vendor_patch(args.orb_dir))
    print(json.dumps({"runner":str(args.runner_dir / "phase2_camera_runner.cc"),
                      "patch_unapplied":str(patch), "contract":CONTRACT}))


def presave_patch(orb):
    """Iterator lifetime fix for the reproducible full-001 native save crash."""
    result=[]
    for relative in ("src/Map.cc","src/MapPoint.cc"):
        original=subprocess.check_output(["git","show",f"{REVISION}:{relative}"],cwd=orb,text=True)
        modified=original
        if relative=="src/Map.cc":
            begin=modified.index("void Map::PreSave(")
            end=modified.index("void Map::PostLoad(",begin)
            body=modified[begin:end]
            assert body.count("for(MapPoint* pMPi : mspMapPoints)")==2
            body=body.replace("    int nMPWithoutObs = 0;", "    const auto pointsToSave = GetAllMapPoints();\n    int nMPWithoutObs = 0;")
            body=body.replace("for(MapPoint* pMPi : mspMapPoints)","for(MapPoint* pMPi : pointsToSave)")
            body=body.replace("                pMPi->EraseObservation(it->first);", "                pMPi->EraseObservation(it->first);\n                if(pMPi->isBad()) break;")
            body=body.replace("        mvpBackupMapPoints.push_back(pMPi);\n        pMPi->PreSave(mspKeyFrames,mspMapPoints);", "        pMPi->PreSave(mspKeyFrames,mspMapPoints);\n        if(!pMPi->isBad()) mvpBackupMapPoints.push_back(pMPi);")
            modified=modified[:begin]+body+modified[end:]
        else:
            old="                mpRefKF=mObservations.begin()->first;"
            assert modified.count(old)==1
            modified=modified.replace(old,"                mpRefKF = mObservations.empty() ? nullptr : mObservations.begin()->first;")
            old="    for(std::map<KeyFrame*,std::tuple<int,int> >::const_iterator it = mObservations.begin(), end = mObservations.end(); it != end; ++it)"
            assert modified.count(old)==1
            modified=modified.replace(old,"    const auto observationsToSave = GetObservations();\n"+old.replace("mObservations.begin()","observationsToSave.begin()").replace("mObservations.end()","observationsToSave.end()"))
            old="            EraseObservation(pKFi);"
            assert modified.count(old)==1
            modified=modified.replace(old,old+"\n            if(isBad()) return;")
        result.extend(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),
                                          fromfile="a/"+relative,tofile="b/"+relative))
    return "".join(result)


GBA_CHECK_CPP = r'''// Exercises the actual native lifecycle helper with a controlled worker.
#include "LoopClosing.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
struct Probe : ORB_SLAM3::LoopClosing {
    Probe():LoopClosing(nullptr,nullptr,nullptr,false,true) {}
    using LoopClosing::StopAndJoinGBA;
    std::atomic<bool> started{false};
    std::atomic<int> published{0};
    void launch(bool wait_for_cancel) {
        StopAndJoinGBA(false);
        std::unique_lock<std::mutex> lock(mMutexGBA);
        started=false; mbStopGBA=false; mbRunningGBA=true; mbFinishedGBA=false;
        const auto generation=++mnFullBAIdx;
        mpThreadGBA=new std::thread([this,generation,wait_for_cancel] {
            started=true;
            if(wait_for_cancel) for(;;) {
                { std::unique_lock<std::mutex> lock(mMutexGBA); if(mbStopGBA) break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::unique_lock<std::mutex> lock(mMutexGBA);
            if(generation==mnFullBAIdx && !mbStopGBA) ++published;
            mbRunningGBA=false; mbFinishedGBA=true;
        });
    }
    bool idle() {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return !mpThreadGBA && !mbRunningGBA && mbFinishedGBA;
    }
};
int main() {
    Probe p;
    for(int i=0;i<3;++i) {
        p.launch(true);
        while(!p.started) std::this_thread::yield();
        p.StopAndJoinGBA(true);
        if(!p.idle() || p.published!=i) throw std::runtime_error("canceled job committed or remained live");
        p.launch(false);
        p.Phase2JoinGBA();
        if(!p.idle() || p.published!=i+1) throw std::runtime_error("completed job was not joined/published");
    }
    std::cout << "GBA lifecycle check passed: 3 canceled jobs, 3 completed jobs, no join-under-lock deadlock\n";
}
'''


def gba_patch(orb):
    """Patch the diagnosed native baseline, after diagnostics/presave/native build fixes."""
    paths=["include/LoopClosing.h","src/LoopClosing.cc"]
    before={p:(orb / p).read_text() for p in paths}
    header,source=(before[p] for p in paths)
    if "StopAndJoinGBA" in header or "unsigned long mnFullBAIdx;" not in header:
        raise ValueError("Expected the pre-lifecycle baseline with unsigned long GBA generation")
    header=header.replace("unsigned long nLoopKF);","unsigned long nLoopKF, unsigned long generation);")
    start=header.index("    void Phase2JoinGBA() {")
    end=header.index("\n\n    bool isFinished();",start)
    header=header[:start]+"    void Phase2JoinGBA() { StopAndJoinGBA(false); }"+header[end:]
    header=header.replace("    std::vector<std::thread*> mPhase2RetiredGBA;", "    void StopAndJoinGBA(bool cancel);\n    void LaunchGBA(Map* map, unsigned long keyframeId);")
    # There are exactly three cancellation sites; preserve visual/inertial relaunch decisions.
    assert source.count("    if(isRunningGBA())\n    {")==3
    for _ in range(3):
        start=source.index("    if(isRunningGBA())\n    {")
        pos=source.index("{",start); depth=1; end=pos+1
        while depth:
            depth += (source[end]=="{")-(source[end]=="}"); end+=1
        old=source[start:end]
        replacement=("    bRelaunchBA = isRunningGBA();\n" if "bRelaunchBA = true" in old else "")+"    StopAndJoinGBA(true);"
        source=source[:start]+replacement+source[end:]
    old="""    mpLocalMapper->RequestStop();
    mpLocalMapper->EmptyQueue(); // Proccess keyframes in the queue

    // If a Global Bundle Adjustment is running, abort it
    StopAndJoinGBA(true);"""
    assert source.count(old)==1
    source=source.replace(old,"""    // A completed GBA may release LocalMapping; reap it before requesting stop.
    StopAndJoinGBA(true);
    mpLocalMapper->RequestStop();""")
    anchor='    // Ensure current keyframe is updated\n    //cout << "Start updating connections" << endl;'
    assert source.count(anchor)==1
    source=source.replace(anchor,"    mpLocalMapper->EmptyQueue(); // LocalMapping is now quiescent.\n\n"+anchor)
    source=source.replace("while(!mpLocalMapper->isStopped())", "while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())")
    old="""        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mnCorrectionGBA = mnNumCorrection;

        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, pLoopMap, mpCurrentKF->mnId);"""
    assert source.count(old)==1
    source=source.replace(old,"        mnCorrectionGBA = mnNumCorrection;\n        LaunchGBA(pLoopMap, mpCurrentKF->mnId);")
    old="""        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this, pMergeMap, mpCurrentKF->mnId);"""
    assert source.count(old)==1
    source=source.replace(old,"        LaunchGBA(pMergeMap, mpCurrentKF->mnId);")
    signature="void LoopClosing::RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF)"
    assert source.count(signature)==1
    source=source.replace(signature,"""void LoopClosing::StopAndJoinGBA(bool cancel)
{
    std::thread* worker;
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(cancel) { mbStopGBA = true; ++mnFullBAIdx; }
        worker = mpThreadGBA;
        mpThreadGBA = nullptr;
    }
    // The worker takes mMutexGBA to publish/exit: never hold it while joining.
    if(worker) { if(worker->joinable()) worker->join(); delete worker; }
    unique_lock<mutex> lock(mMutexGBA);
    mbRunningGBA = false;
    mbFinishedGBA = true;
}

void LoopClosing::LaunchGBA(Map* map, unsigned long keyframeId)
{
    StopAndJoinGBA(false); // Reap normally completed jobs as well as canceled jobs.
    unique_lock<mutex> lock(mMutexGBA);
    mbStopGBA = false;
    mbRunningGBA = true;
    mbFinishedGBA = false;
    const unsigned long generation = ++mnFullBAIdx;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, map, keyframeId, generation);
}

void LoopClosing::RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF, unsigned long generation)""")
    assert source.count("    unsigned long idx = mnFullBAIdx;")==1
    source=source.replace("    unsigned long idx = mnFullBAIdx;\n","")
    old="""        if(idx!=mnFullBAIdx)
            return;

        if(!bImuInit && pActiveMap->isImuInitialized())
            return;"""
    assert source.count(old)==1
    source=source.replace(old,"""        if(generation!=mnFullBAIdx || (!bImuInit && pActiveMap->isImuInitialized()))
        {
            mbRunningGBA = false;
            mbFinishedGBA = true;
            return;
        }""")
    assert "mPhase2RetiredGBA" not in source+header and "->detach()" not in source
    return "".join(line for p,after in zip(paths,[header,source])
                   for line in difflib.unified_diff(before[p].splitlines(True),after.splitlines(True),
                                                    fromfile="a/"+p,tofile="b/"+p))


def cross_session_proof(loaded, final, frames, *, requested, disjoint, complete):
    old={kf["id"] for m in loaded["maps"] for kf in m["keyframes"]}
    crossing=[]
    for m in final["maps"]:
        members={kf["id"] for kf in m["keyframes"]}
        for kf in m["keyframes"]:
            for edge in kf["merge_edges"]:
                if edge in members and ((kf["id"] in old) != (edge in old)):
                    pair=sorted([kf["id"],edge])
                    if pair not in [e["keyframe_ids"] for e in crossing]:
                        crossing.append({"map_id":m["map_id"],"keyframe_ids":pair})
    continuing=[f["source_index"] for f in frames if f["valid"] and
                f["cross_session_merge_edge_observed"] and f["loaded_map_points_tracked"]>0]
    passed=bool(requested and complete and disjoint and old and crossing and continuing)
    return {"requested":requested,"passed":passed,"loaded_keyframes":len(old),
            "source_frames_disjoint_from_prior_sessions":disjoint,
            "new_edges_connecting_loaded_and_new_keyframes":crossing,
            "valid_frames_tracking_loaded_landmarks_after_merge":continuing,
            "rule":"Separate process + disjoint source frames + completed merge edge connecting loaded/new IDs in one live map + subsequent OK tracking of loaded landmarks. File load or map ID alone never passes."}


def read_euroc(path):
    """Return 8-column string rows; collapse only exactly identical repeated records."""
    raw=[]
    for line in path.read_text().splitlines():
        row=line.split()
        if len(row)!=8 or not np.isfinite(np.array(row,float)).all():
            raise ValueError("Malformed final EuRoC trajectory")
        if raw and float(row[0])==float(raw[-1][0]):
            if row!=raw[-1]: raise ValueError("Conflicting final poses at one timestamp")
            continue # Native RECENTLY_LOST storage may repeat the previous record.
        if raw and float(row[0])<float(raw[-1][0]):
            raise ValueError("Nonmonotonic final EuRoC trajectory")
        raw.append(row)
    return raw


def evaluate(dataset, output, frames, sensor="monocular"):
    """Called after subprocess exit; never feeds alignment or truth to the runner."""
    gt=read_rows(dataset / "groundtruth.txt",8)
    raw=read_euroc(output / "trajectory.final.euroc.txt")
    if not raw: return {"status":"no_final_poses"}
    times=np.array([float(r[0])/1e9 for r in raw])
    good=[f for f in frames if f["valid"]]
    frame_pairs=associate([f["timestamp"] for f in good],times,0.000002)
    chosen=[j for _,j in frame_pairs]
    gt_pairs=associate(times[chosen],[float(r[0]) for r in gt],0.02)
    if len(gt_pairs)<3:
        return {"status":"insufficient_valid_final_poses", "matched_poses":len(gt_pairs)}
    x=np.array([raw[chosen[i]][1:4] for i,_ in gt_pairs],float)
    y=np.array([gt[j][1:4] for _,j in gt_pairs],float)
    se3=metric_ate(x,y)
    centered=x-x.mean(0)
    variance=float(np.sum(centered**2))
    if variance<=1e-12:
        return {"status":"degenerate_trajectory", "matched_poses":len(x)}
    u,s,vt=np.linalg.svd(centered.T @ (y-y.mean(0)))
    correction=np.ones(3); correction[-1]=np.linalg.det(vt.T @ u.T)
    scale=float(np.sum(s*correction)/variance)
    if not np.isfinite(scale) or scale<=0:
        raise ValueError("Invalid fitted similarity scale")
    sim3=metric_ate(scale*x,y)
    sim3.update(alignment="Sim3 similarity, posthoc evaluation only",scale_fit=True,scale=scale)
    return {"status":"evaluation_complete", "evaluation_only":True,
            "groundtruth_sha256":digest(dataset / "groundtruth.txt"),
            "groundtruth_entered_reconstruction":False,"native_outputs_rescaled":False,
            "trajectory":"final EuRoC rebased camera poses, filtered by exact input state==2 ledger",
            "sensor_depth_entered_reconstruction":sensor=="rgbd",
            "units_note":("Sensor-depth RGB-D positions are meters; primary SE3 ATE fixes scale to 1." if sensor=="rgbd" else "ORB monocular units are uncalibrated. Scale-1 error compares native coordinates numerically with meter GT; it does not establish metric reconstruction."),
            "primary_metric":"SE3_scale_1_meters" if sensor=="rgbd" else "Sim3_fitted_meters_per_native_unit_diagnostic",
            ("SE3_scale_1_meters" if sensor=="rgbd" else "SE3_scale_1_native_units_diagnostic"):se3,
            "Sim3_fitted_meters_per_native_unit_diagnostic":sim3,
            "positions_rank":int(np.linalg.matrix_rank(centered)),
            "associations":[{"rgb_timestamp":float(times[chosen[i]]),"gt_timestamp":float(gt[j][0]),
                             "delta_s":float(abs(times[chosen[i]]-float(gt[j][0])))} for i,j in gt_pairs]}


def run(args):
    dataset,output=args.dataset.resolve(),args.output.resolve()
    binary,orb=args.binary.resolve(),args.orb_dir.resolve()
    if subprocess.check_output([str(binary),"--contract"],text=True).strip()!=CONTRACT:
        raise ValueError("Runner contract mismatch")
    rows=read_rows(dataset / "rgb.txt",2)
    stop=len(rows) if args.stop is None else args.stop
    if not 0<=args.start<stop<=len(rows):
        raise ValueError("Expected 0 <= start < stop <= RGB frame count")
    rgbd=args.sensor=="rgbd"
    units="sensor_depth_meters" if rgbd else "uncalibrated_monocular"
    depth_rows=read_rows(dataset / "depth.txt",2) if rgbd else []
    pairs=associate([float(r[0]) for r in rows],[float(d[0]) for d in depth_rows],0.02) if rgbd else []
    depth_for_rgb=dict(pairs)
    pairing={"strict_maximum_delta_s":0.02,"one_to_one":True,"rgb_frames":len(rows),
             "depth_frames":len(depth_rows),"paired_frames":len(pairs),
             "unmatched_rgb_frames":len(rows)-len(pairs),"unmatched_depth_frames":len(depth_rows)-len(pairs)} if rgbd else None
    selected=[]
    for index in range(args.start,stop):
        if rgbd and index not in depth_for_rgb: continue
        stamp,relative=rows[index]
        source=(dataset / relative).resolve()
        if not source.is_relative_to(dataset) or not source.is_file() or any(c in str(source) for c in "\t\r\n"):
            raise ValueError(f"Missing or unsafe image: {source}")
        selected.append({"source_index":index,"timestamp":float(stamp),"timestamp_text":stamp,
                         "source_path":str(source),"sha256":digest(source)})
        if rgbd:
            depth_stamp,depth_relative=depth_rows[depth_for_rgb[index]]
            depth_source=(dataset / depth_relative).resolve()
            if not depth_source.is_relative_to(dataset) or not depth_source.is_file() or any(c in str(depth_source) for c in "\t\r\n"):
                raise ValueError(f"Missing or unsafe depth: {depth_source}")
            selected[-1].update(depth_source_path=str(depth_source),depth_timestamp=float(depth_stamp),
                                depth_timestamp_text=depth_stamp,depth_sha256=digest(depth_source),
                                depth_delta_s=abs(float(depth_stamp)-float(stamp)))
    if not selected: raise ValueError("No selected sensor inputs")
    if bool(args.mask_manifest) != bool(args.video_manifest):
        raise ValueError("--mask-manifest requires --video-manifest and vice versa")
    output.mkdir(parents=True,exist_ok=False)
    mask_provenance = prepare_allow_masks(args.mask_manifest, args.video_manifest, dataset, rows, selected, output) if args.mask_manifest else None
    native_patch=output / "native-source.patch"
    native_patch.write_bytes(subprocess.check_output(["git","diff"],cwd=orb))
    vocab=(orb / "Vocabulary/ORBvoc.txt").resolve()
    if not vocab.is_file(): raise ValueError(f"Vocabulary missing: {vocab}")
    camera_folder="RGB-D" if rgbd else "Monocular"
    config=(orb / f"Examples/{camera_folder}/{args.camera.upper()}.yaml").read_text()
    if config.count("Camera.RGB: 1")!=1: raise ValueError("Unexpected upstream camera config")
    config=config.replace("Camera.RGB: 1","Camera.RGB: 0")
    if rgbd:
        factor=re.search(r"^RGBD.DepthMapFactor:\s*([\d.]+)",config,re.MULTILINE)
        if not factor or float(factor[1])!=5000: raise ValueError("Expected official TUM depth factor 5000")
    config+='\nSystem.SaveAtlasToFile: "atlas"\n'
    prior_sources=[]
    prior_hashes=[]
    prior=None
    if args.load_run:
        prior_path=args.load_run.resolve()
        prior=json.loads((prior_path / "run.json").read_text())
        if not prior.get("execution_complete"):
            raise ValueError("Prior session did not finish cleanly")
        if prior["native_scale"]!=units: raise ValueError("Atlas and new sensor use different scale conventions")
        old_atlas=prior_path / "atlas.osa"
        if digest(old_atlas)!=prior["artifacts_sha256"]["atlas.osa"]:
            raise ValueError("Prior Atlas changed")
        shutil.copyfile(old_atlas,output / "input.osa")
        config+='System.LoadAtlasFromFile: "input"\n'
        prior_sources=prior["all_session_source_paths"]
        prior_hashes=prior["all_session_source_sha256"]
    (output / "camera.yaml").write_text(config)
    (output / "input.tsv").write_text("".join(f"{f['source_index']}\t{f['timestamp_text']}\t{f['source_path']}"+
                                             (f"\t{f['depth_timestamp_text']}\t{f['depth_source_path']}" if rgbd else "")+
                                             (f"\t{f['allow_mask_path']}" if args.mask_manifest else "")+"\n" for f in selected))
    write_json(output / "input.manifest.json",selected)
    source_paths=[f["source_path"] for f in selected]
    metadata={"status":"running","contract":CONTRACT,"dataset":str(dataset),
              "rgb_index_sha256":digest(dataset / "rgb.txt"),"source_revision":REVISION,
              "camera":args.camera,"color_order":"BGR", "source_resolution_wh":[640,480],
              "pixel_transform":"identity; original distorted FR1 pixels or already-undistorted FR3 pixels",
              "sensor":args.sensor,"sensor_depth_entered_reconstruction":rgbd,
              "groundtruth_entered_reconstruction":False,"native_scale":units,
              "dynamic_mask_applied":bool(args.mask_manifest),"mask_provenance":mask_provenance,"rgb_depth_pairing":pairing,
              "depth_index_sha256":digest(dataset / "depth.txt") if rgbd else None,
              "depth_map_factor":5000 if rgbd else None,
              "online_pose_convention":"c2w in native map coordinates at tracking time; may change after loop/merge/BA; map ID is diagnostic, not an immutable coordinate ID",
              "final_pose_convention":"EuRoC c2w, nanosecond timestamps, first keyframe origin; trajectory.export.json maps these poses back to final Atlas coordinates",
              "binary_sha256":digest(binary),"vocabulary_sha256":digest(vocab),
              "libORB_SLAM3_sha256":digest(orb / "lib/libORB_SLAM3.dylib"),
              "native_source_patch_sha256":digest(native_patch),
              "settings_sha256":digest(output / "camera.yaml"),
              "source_frame_count":len(rows),"selected_frame_count":len(selected),
              "unmatched_rgb_frames_in_selected_range":stop-args.start-len(selected),
              "pacing":"Official mono_tum next source interval (last uses preceding; singleton zero), minus only Track wall time; diagnostic overhead is additional",
              "selected_range_start_stop":[args.start,stop],"load_run":str(args.load_run) if args.load_run else None,
              "loaded_atlas_sha256":digest(output / "input.osa") if args.load_run else None,
              "all_session_source_paths":sorted(set(prior_sources+source_paths)),
              "all_session_source_sha256":sorted(set(prior_hashes+[f["sha256"] for f in selected]))}
    write_json(output / "run.json",metadata)
    started=time.monotonic()
    command=[str(binary),str(vocab),str(output / "camera.yaml"),str(output / "input.tsv"),args.sensor,
             "allow-mask" if args.mask_manifest else "none"]
    with (output / "runner.log").open("w") as log:
        try:
            process=subprocess.run(command,cwd=output,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            returncode=process.returncode
        except subprocess.TimeoutExpired:
            returncode=124
    metadata.update(process_returncode=returncode,elapsed_seconds=time.monotonic()-started,command=command)
    write_json(output / "run.json",metadata)
    frames=[]
    ledger=output / "frames.online.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try: frames.append(json.loads(line))
            except json.JSONDecodeError: break
    expected=[(f["source_index"],f["timestamp"]) for f in selected]
    observed=[(f["source_index"],f["timestamp"]) for f in frames]
    complete=returncode==0 and observed==expected and (output / "atlas.osa").is_file()
    for frame in frames:
        if frame["valid"] != (frame["state"]==2) or ((frame["c2w_online_native"] is not None)!=frame["valid"]):
            raise ValueError("Runner emitted misleading pose validity")
    if args.mask_manifest:
        by_index = {f["source_index"]: f for f in selected}
        for frame in frames:
            source = by_index[frame["source_index"]]
            if (not frame["dynamic_mask_applied"] or frame["excluded_region_keypoints"] != 0
                    or frame["allow_mask_pixels"] != source["allow_mask_pixels"]
                    or digest(Path(source["allow_mask_path"])) != source["allow_mask_sha256"]):
                raise ValueError("Native feature masking or mask provenance disagrees with input")
    loaded=json.loads((output / "atlas.loaded.json").read_text()) if (output / "atlas.loaded.json").exists() else {"maps":[]}
    final=json.loads((output / "atlas.final.json").read_text()) if (output / "atlas.final.json").exists() else {"maps":[]}
    proof=cross_session_proof(loaded,final,frames,requested=bool(args.load_run),
                              disjoint=not set(prior_hashes).intersection(f["sha256"] for f in selected),complete=complete)
    metadata.update(status="execution_complete" if complete else "execution_failed",execution_complete=complete,
                    processed_frames=len(frames),states=dict(Counter(f["state"] for f in frames)),
                    valid_pose_count=sum(f["valid"] for f in frames),
                    valid_pose_fraction=sum(f["valid"] for f in frames)/len(selected),atlas_reuse=proof,
                    valid_source_rgb_fraction=sum(f["valid"] for f in frames)/(stop-args.start),
                    m1_complete=False,m1_note="This camera experiment does not deliver the required static surface mesh or full A→B→A product acceptance.")
    if complete and (output / "trajectory.final.euroc.txt").stat().st_size:
        try: evaluation=evaluate(dataset,output,frames,args.sensor)
        except (ValueError,OSError) as error: evaluation={"status":"evaluation_failed","reason":str(error)}
        write_json(output / "evaluation.json",evaluation)
        metadata["evaluation_status"]=evaluation["status"]
    metadata["artifacts_sha256"]={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!="run.json"}
    write_json(output / "run.json",metadata)
    print(json.dumps({"output":str(output),"execution_complete":complete,"atlas_reuse_passed":proof["passed"],
                      "valid_pose_fraction":metadata["valid_pose_fraction"]}))
    return 0 if complete and (not args.load_run or proof["passed"]) else 2


def self_check():
    loaded={"maps":[{"keyframes":[{"id":1,"merge_edges":[]}]}]}
    final={"maps":[{"map_id":99,"keyframes":[{"id":1,"merge_edges":[2]},{"id":2,"merge_edges":[1]}]}]}
    frame={"source_index":9,"valid":True,"cross_session_merge_edge_observed":True,"loaded_map_points_tracked":3}
    kwargs=dict(requested=True,disjoint=True,complete=True)
    assert cross_session_proof(loaded,final,[frame],**kwargs)["passed"]
    for key in kwargs:
        assert not cross_session_proof(loaded,final,[frame],**{**kwargs,key:False})["passed"]
    assert not cross_session_proof(loaded,{"maps":[]},[frame],**kwargs)["passed"]
    assert not cross_session_proof(loaded,final,[],**kwargs)["passed"]
    assert not cross_session_proof(loaded,final,[{**frame,"loaded_map_points_tracked":0}],**kwargs)["passed"]
    # End-to-end independent synthetic truth: known scale 3, rigid transform,
    # one RECENTLY_LOST record must not enter the final trajectory evaluation.
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[100,100,100]],float)
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        y=3*x @ rotation.T+[4,5,6]
        for name,values,factor in [("groundtruth.txt",y,1),("trajectory.final.euroc.txt",x,1e9)]:
            (root / name).write_text("".join(f"{(10+i)*factor:.0f} {p[0]} {p[1]} {p[2]} 0 0 0 1\n" for i,p in enumerate(values)))
        ledger=[{"timestamp":10+i,"valid":i<4} for i in range(5)]
        result=evaluate(root,root,ledger)
        assert result["Sim3_fitted_meters_per_native_unit_diagnostic"]["matched_poses"]==4
        assert abs(result["Sim3_fitted_meters_per_native_unit_diagnostic"]["scale"]-3)<1e-10
        assert result["Sim3_fitted_meters_per_native_unit_diagnostic"]["rmse_m"]<1e-10
        assert result["SE3_scale_1_native_units_diagnostic"]["rmse_m"]>1
        sensor_result=evaluate(root,root,ledger,"rgbd")
        assert sensor_result["primary_metric"]=="SE3_scale_1_meters"
        assert sensor_result["SE3_scale_1_meters"]["scale"]==1 and not sensor_result["SE3_scale_1_meters"]["scale_fit"]
        row="1000000000 1 2 3 0 0 0 1\n"
        test=root / "duplicates.euroc.txt"
        test.write_text(row+row)
        assert len(read_euroc(test))==1
        test.write_text(row+row.replace("1 2 3","9 2 3"))
        try: read_euroc(test)
        except ValueError: pass
        else: raise AssertionError("Conflicting duplicate pose accepted")
    # Exercise the real media/RLE/provenance boundary with a complete tiny clip.
    import cv2
    from build_video_pose_preview import decode_coco_rle
    from ehs_spatial.providers.sam3 import encode_coco_rle
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory).resolve(); dataset=root / "dataset"; dataset.mkdir()
        video=root / "source.mp4"
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"avc1"),25,(640,480))
        assert writer.isOpened()
        image=np.zeros((480,640,3),np.uint8)
        rows=[]
        for i in range(2):
            cv2.imwrite(str(dataset / f"{i}.png"),image)
            writer.write(image); rows.append([str(10+i),f"{i}.png"])
        writer.release()
        (dataset / "rgb.txt").write_text("10 0.png\n11 1.png\n")
        mask=np.zeros((480,640),bool); mask[10:20,30:40]=True
        obj={"label":"person","rle":encode_coco_rle(mask)}
        raw={"input":{"sha256":digest(video),"frame_count":2,"height":480,"width":640,
                      "duration_seconds":.08,"frame_timestamps_seconds":[0,.04]},
             "frames":[{"frame_index":i,"source_frame_index":i,"timestamp_seconds":i*.04,
                        "objects":[obj,obj] if i==0 else []} for i in range(2)]}
        raw_path=root / "observations.json"; write_json(raw_path,raw)
        write_json(root / "run.json",{"status":"execution_complete","raw_sha256":digest(raw_path)})
        write_json(root / "source-manifest.json",{"sourceVideo":str(video),"sourceSha256":digest(video),
                   "sourceStartFrame":0,"frameCount":2,"clipSha256":digest(video),
                   "sourceFrameMediaTimesSeconds":[0,.04,.08]})
        video_manifest=root / "video-manifest.json"
        mapping="MP4 frame i corresponds to data row i of original rgb.txt (zero-based, excluding comment lines). Every RGB input frame appears once and in order."
        write_json(video_manifest,{"original_root":str(dataset),"playback":{"sha256":digest(video),
                   "frame_mapping":mapping,"input_frames":2,"decoded_frames":2},
                   "streams":{"rgb":{"count":2}},
                   "original_files":[{"path":f.name,"sha256":digest(f)} for f in dataset.iterdir()]})
        selected=[{"source_index":i} for i in range(2)]
        prepare_allow_masks(raw_path,video_manifest,dataset,rows,selected,root)
        first=cv2.imread(selected[0]["allow_mask_path"],cv2.IMREAD_UNCHANGED)
        assert np.array_equal(first==0,mask) and selected[0]["overlapping_instance_pixels"]==100
        assert selected[0]["excluded_pixels"]==100 and selected[1]["allow_mask_pixels"]==480*640
        assert np.array_equal(decode_coco_rle(obj["rle"]).astype(bool),mask)
        raw["frames"].pop(); write_json(raw_path,raw)
        write_json(root / "run.json",{"status":"execution_complete","raw_sha256":digest(raw_path)})
        try: prepare_allow_masks(raw_path,video_manifest,dataset,rows,selected,root)
        except ValueError as error: assert "every original RGB frame" in str(error)
        else: raise AssertionError("Incomplete masks accepted")
    print("phase2 camera self-check passed: provenance gates, false merge cases, state filtering, SE3/Sim3, native mask union and missing frames")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    sub.add_parser("self-check")
    prep=sub.add_parser("prepare")
    prep.add_argument("--orb-dir",type=Path,default=DEFAULT_VENDOR / "ORB_SLAM3")
    prep.add_argument("--runner-dir",type=Path,default=DEFAULT_VENDOR / "runner")
    save_fix=sub.add_parser("prepare-presave")
    save_fix.add_argument("--orb-dir",type=Path,default=DEFAULT_VENDOR / "ORB_SLAM3")
    save_fix.add_argument("--runner-dir",type=Path,default=DEFAULT_VENDOR / "runner")
    gba_fix=sub.add_parser("prepare-gba")
    gba_fix.add_argument("--orb-dir",type=Path,default=DEFAULT_VENDOR / "ORB_SLAM3")
    gba_fix.add_argument("--runner-dir",type=Path,default=DEFAULT_VENDOR / "runner")
    mask_fix=sub.add_parser("prepare-mask")
    mask_fix.add_argument("--orb-dir",type=Path,default=DEFAULT_VENDOR / "ORB_SLAM3")
    mask_fix.add_argument("--runner-dir",type=Path,default=DEFAULT_VENDOR / "runner")
    execute=sub.add_parser("run")
    execute.add_argument("--orb-dir",type=Path,default=DEFAULT_VENDOR / "ORB_SLAM3")
    execute.add_argument("--binary",type=Path,required=True)
    execute.add_argument("--dataset",type=Path,required=True)
    execute.add_argument("--output",type=Path,required=True)
    execute.add_argument("--camera",choices=["tum1","tum3"],default="tum1")
    execute.add_argument("--sensor",choices=["monocular","rgbd"],default="monocular")
    execute.add_argument("--mask-manifest",type=Path,help="Completed SAM observations.json; every original RGB frame is required")
    execute.add_argument("--video-manifest",type=Path,help="Verified source video to original RGB mapping")
    execute.add_argument("--start",type=int,default=0)
    execute.add_argument("--stop",type=int)
    execute.add_argument("--load-run",type=Path)
    execute.add_argument("--timeout",type=float,default=1800)
    args=parser.parse_args()
    if args.mode=="prepare": prepare(args)
    elif args.mode=="prepare-presave":
        args.runner_dir.mkdir(parents=True,exist_ok=True)
        target=args.runner_dir / "phase2-camera-presave.patch"
        target.write_text(presave_patch(args.orb_dir))
        print(f"Unapplied serialization fix: {target}")
    elif args.mode=="prepare-gba":
        args.runner_dir.mkdir(parents=True,exist_ok=True)
        target=args.runner_dir / "phase2-camera-gba.patch"
        target.write_text(gba_patch(args.orb_dir))
        (args.runner_dir / "phase2_camera_gba_check.cc").write_text(GBA_CHECK_CPP)
        print(f"Unapplied GBA lifecycle fix: {target}")
    elif args.mode=="prepare-mask":
        args.runner_dir.mkdir(parents=True,exist_ok=True)
        target=args.runner_dir / "phase2-camera-mask.patch"
        target.write_text(mask_patch(args.orb_dir))
        (args.runner_dir / "phase2_camera_runner.cc").write_text(CPP)
        print(f"Unapplied feature-mask plumbing: {target}")
    elif args.mode=="self-check": self_check()
    else: return run(args)
    return 0


if __name__=="__main__":
    sys.exit(main())
