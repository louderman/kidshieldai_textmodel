// Background service worker
// Runs persistently in the background and can listen to browser events

chrome.runtime.onInstalled.addListener(() => {
  console.log('KidShield AI installed.')
})
