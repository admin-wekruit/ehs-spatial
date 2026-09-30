import { createContext, useContext } from "react";
import { resolveAsset } from "./api";

// The same report consumes published API assets or local oneshot artifacts.
export const SceneResources = createContext({ resolveAsset, analysisAvailable: true });
export const useSceneResources = () => useContext(SceneResources);
