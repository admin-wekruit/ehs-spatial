// Message ids per identity decision and per measurement key (the texts live in src/locales).
export const identityDecisionKeys: Record<string, string> = {same:'identitySame',different:'identityDifferent',undecided:'identityUndecided'};
export const measurementLabelKeys: Record<string, string> = {
  coordinateFrameId:'identityMeasurementFrame', dimensionsNative:'identityMeasurementDimensions',
  widthNative:'identityMeasurementWidth', depthNative:'identityMeasurementDepth', groundHeightNative:'identityMeasurementHeight',
  basis:'identityMeasurementBasis', projectedHull:'identityMeasurementHull', observedBounds:'identityMeasurementBounds',
  nativeToFloor:'identityMeasurementPlane', groundNormal:'identityMeasurementNormal',
};
