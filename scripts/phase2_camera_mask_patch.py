"""Generate the shared ORB extractor mask implementation and native regression probe.

This emits an unapplied patch against the pinned extractor source. 255 permits
static image support; zero excludes it. Input images remain unchanged.
"""
import argparse
import difflib
from pathlib import Path
import subprocess


MASK_PYRAMID = r'''
    vector<Mat> ORBextractor::ComputeMaskPyramid(const Mat& mask)
    {
        vector<Mat> centers(nlevels);
        if(mask.empty()) return centers;
        if(mask.type()!=CV_8UC1 || mask.size()!=mvImagePyramid[0].size())
            CV_Error(cv::Error::StsBadArg, "ORB allow-mask must be uint8 and match the input image");

        // A rotated pattern sample is bounded by its Euclidean radius. Gaussian
        // preprocessing contributes another 3 pixels (7x7 kernel); orientation
        // and FAST have smaller support. Derive this from the actual pattern.
        int radius=HALF_PATCH_SIZE;
        for(const auto& p : pattern)
            radius=std::max(radius,cvCeil(std::sqrt(double(p.x*p.x+p.y*p.y))));
        radius+=3;
        const Mat kernel=getStructuringElement(MORPH_RECT,Size(2*radius+1,2*radius+1));
        Mat unsafe, unsafe8;
        compare(mask,0,unsafe8,CMP_EQ);
        unsafe8.convertTo(unsafe,CV_32F,1.0/255.0);
        for(int level=0;level<nlevels;++level)
        {
            // Follow the image pyramid's sequential INTER_LINEAR sampling.
            // Float support prevents a small excluded contribution rounding to 0.
            if(level) {
                Mat next;
                resize(unsafe,next,mvImagePyramid[level].size(),0,0,INTER_LINEAR);
                unsafe=next;
            }
            Mat allowed;
            compare(unsafe,0,allowed,CMP_EQ);
            erode(allowed,centers[level],kernel,Point(-1,-1),1,BORDER_REFLECT_101);
        }
        return centers;
    }

'''


CHECK_CPP = r'''// Counterfactual regression: changing only excluded pixels must not
// change selected features or descriptors at any pyramid level.
#include "ORBextractor.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdexcept>

struct Result { std::vector<cv::KeyPoint> keys; cv::Mat descriptors; };
static Result extract(ORB_SLAM3::ORBextractor& orb,const cv::Mat& image,const cv::Mat& mask) {
    Result result; std::vector<int> overlap{0,0};
    orb(image,mask,result.keys,result.descriptors,overlap); return result;
}
static void same(const Result& a,const Result& b) {
    if(a.keys.size()!=b.keys.size()) throw std::runtime_error("feature count changed");
    for(size_t i=0;i<a.keys.size();++i) {
        const auto& x=a.keys[i]; const auto& y=b.keys[i];
        if(x.pt!=y.pt || x.angle!=y.angle || x.octave!=y.octave || x.response!=y.response)
            throw std::runtime_error("excluded pixels changed feature selection/orientation");
    }
    if(a.descriptors.size()!=b.descriptors.size() || cv::norm(a.descriptors,b.descriptors,cv::NORM_INF)!=0)
        throw std::runtime_error("excluded pixels changed a descriptor");
}
int main() {
    ORB_SLAM3::ORBextractor orb(1000,1.2,8,20,7);
    cv::Mat image(480,640,CV_8UC1),all(480,640,CV_8UC1,cv::Scalar(255));
    cv::RNG rng(48173); rng.fill(image,cv::RNG::UNIFORM,0,256);
    const cv::Mat original=image.clone();
    const auto noMask=extract(orb,image,cv::Mat());
    same(noMask,extract(orb,image,all));
    if(noMask.keys.size()<500) throw std::runtime_error("insufficient test texture");
    cv::Mat allow=all.clone();
    allow(cv::Rect(150,80,230,290)).setTo(0);
    allow(cv::Rect(0,0,37,480)).setTo(0); // Exercise image and pyramid borders.
    cv::Mat changed=image.clone();
    changed.setTo(0,allow==0);
    const auto masked=extract(orb,image,allow);
    same(masked,extract(orb,changed,allow));
    if(masked.keys.empty()) throw std::runtime_error("mask incorrectly removed all texture");
    bool levels[8]={false};
    for(const auto& k:masked.keys) {
        if(!allow.at<unsigned char>(cvRound(k.pt.y),cvRound(k.pt.x)))
            throw std::runtime_error("keypoint is inside excluded foreground");
        levels[k.octave]=true;
    }
    for(bool present:levels) if(!present) throw std::runtime_error("pyramid level untested");
    cv::Mat blocked(480,640,CV_8UC1,cv::Scalar(0));
    const auto none=extract(orb,image,blocked);
    if(!none.keys.empty() || !none.descriptors.empty()) throw std::runtime_error("all-excluded mask leaked features");
    bool rejected=false;
    try { extract(orb,image,cv::Mat(240,320,CV_8UC1)); } catch(const cv::Exception&) { rejected=true; }
    if(!rejected) throw std::runtime_error("mask dimensions were not checked");
    if(cv::norm(original,image,cv::NORM_INF)!=0) throw std::runtime_error("input image was changed");
    std::cout<<"ORB mask check passed: all 8 levels, excluded-pixel counterfactual invariance, all/none masks, size rejection; "
             <<masked.keys.size()<<" static features"<<std::endl;
}
'''


def generate(orb, directory):
    directory.mkdir(parents=True, exist_ok=True)
    diff = []
    for relative in ["include/ORBextractor.h", "src/ORBextractor.cc"]:
        original = subprocess.check_output(["git", "-C", str(orb), "show", f"HEAD:{relative}"], text=True)
        text = original
        if relative.endswith(".h"):
            text = text.replace("// Mask is ignored in the current implementation.",
                                "// Nonzero mask permits static image support; zero excludes foreground.")
            old = "    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);"
            assert text.count(old) == 1
            text = text.replace(old, "    std::vector<cv::Mat> ComputeMaskPyramid(const cv::Mat& mask);\n"
                                "    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints, const std::vector<cv::Mat>& masks);")
        else:
            old = "    void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)"
            assert text.count(old) == 1
            text = text.replace(old, MASK_PYRAMID + old[:-1] + ", const vector<Mat>& masks)")
            begin, end = text.index("    void ORBextractor::ComputeKeyPointsOctTree"), text.index("    void ORBextractor::ComputeKeyPointsOld")
            body = text[begin:end]
            old = "                    vector<cv::KeyPoint> vKeysCell;"
            assert body.count(old) == 1
            body = body.replace(old, old + r'''
                    const auto filterCandidates = [&](vector<KeyPoint>& keys) {
                        if(masks[level].empty()) return;
                        keys.erase(std::remove_if(keys.begin(),keys.end(),[&](const KeyPoint& key) {
                            return !masks[level].at<uchar>(cvRound(key.pt.y+iniY),cvRound(key.pt.x+iniX));
                        }),keys.end());
                    };
''')
            for threshold in ["iniThFAST", "minThFAST"]:
                needle = "vKeysCell," + threshold + ",true);"
                assert body.count(needle) == 2 # The second occurrence is commented upstream code.
                body = body.replace(needle, needle + "\n                    filterCandidates(vKeysCell);", 1)
            text = text[:begin] + body + text[end:]
            old = "        ComputeKeyPointsOctTree(allKeypoints);"
            assert text.count(old) == 1
            text = text.replace(old, "        ComputeKeyPointsOctTree(allKeypoints, ComputeMaskPyramid(_mask.getMat()));")
        diff.extend(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                         fromfile="a/" + relative, tofile="b/" + relative))
    patch = directory / "phase2-camera-extractor-mask.patch"
    patch.write_text("".join(diff))
    (directory / "phase2_camera_mask_check.cc").write_text(CHECK_CPP)
    print(patch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orb-dir", type=Path, required=True)
    parser.add_argument("--runner-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.orb_dir, args.runner_dir)
