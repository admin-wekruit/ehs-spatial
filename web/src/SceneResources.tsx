import { createContext, useContext } from "react";
import { resolveAsset } from "./api";
import type { BendAnalysis, InclinationAnalysis } from "./SpatialMeasurements";

// The same report consumes published API assets or local oneshot artifacts.
export const SceneResources = createContext<{ resolveAsset: typeof resolveAsset; analysisAvailable: boolean; bendAnalysis?: BendAnalysis; inclinationAnalysis?: InclinationAnalysis }>({ resolveAsset, analysisAvailable: true });
export const useSceneResources = () => useContext(SceneResources);
