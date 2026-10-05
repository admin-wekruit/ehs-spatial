import { createContext, useContext } from "react";
import { resolveAsset } from "./api";
import type { BendAnalysis, BendOutcome, InclinationAnalysis } from "./SpatialMeasurements";

// The same report consumes published API assets or local oneshot artifacts.
export const SceneResources = createContext<{ resolveAsset: typeof resolveAsset; analysisAvailable: boolean; bendAnalysis?: BendAnalysis; inclinationAnalysis?: InclinationAnalysis; layerBends?: BendOutcome[] }>({ resolveAsset, analysisAvailable: true });
export const useSceneResources = () => useContext(SceneResources);
