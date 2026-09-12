window.panoptesSiteConfig = {
  apiOrigin: location.origin,
  siteRoot: new URL('/published/', location.href).href,
  workspaceRoot: new URL('/reports', location.href).href,
  workspaceAssets: new URL('/workspace-assets/', location.href).href,
};
