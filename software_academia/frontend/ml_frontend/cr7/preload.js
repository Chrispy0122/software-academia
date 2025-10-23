const { contextBridge } = require('electron');

// Si luego quieres exponer APIs seguras al renderer, hazlo aquí:
contextBridge.exposeInMainWorld('api', {
  // ejemplo: getVersion: () => process.versions.electron
});
