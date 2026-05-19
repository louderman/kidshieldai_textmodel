chrome.runtime.onInstalled.addListener(() => {
  console.log('KidShield AI installed.');
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === 'classify') {
    fetch('http://127.0.0.1:8000/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: message.text }),
    })
      .then(res => res.ok ? res.json() : null)
      .then(data => sendResponse({ label: data?.label ?? null }))
      .catch(() => sendResponse({ label: null }));
    return true; // keep message channel open for async response
  }

  if (message.type === 'classify-image') {
    fetch('http://127.0.0.1:8000/classify-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: message.image }),
    })
      .then(res => res.ok ? res.json() : null)
      .then(data => sendResponse({ label: data?.label ?? null }))
      .catch(() => sendResponse({ label: null }));
    return true;
  }
});
