const API_URL = 'http://127.0.0.1:8000/classify';
const IMAGE_API_URL = 'http://127.0.0.1:8000/classify-image';
const PROCESSED = 'data-kidshield-processed';
const LINK_PROCESSED = 'data-kidshield-link-processed';
const IMG_PROCESSED = 'data-kidshield-img-processed';

function makeBlurSpan(text) {
  const span = document.createElement('span');
  span.textContent = text;
  span.setAttribute(PROCESSED, 'true');
  span.style.cssText = [
    'filter: blur(6px)',
    'cursor: pointer',
    'user-select: none',
    'display: inline',
    'transition: filter 0.2s',
  ].join(';');
  span.title = 'KidShield: unsafe content hidden — click to reveal';
  span.addEventListener('click', () => {
    span.style.filter = 'none';
    span.style.cursor = 'default';
    span.title = '';
  });
  return span;
}

async function classifyNode(textNode) {
  const text = textNode.textContent.trim();
  if (!text || text.length < 2) return;

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) return;
    const { label } = await res.json();
    if (label === 'unsafe') {
      const span = makeBlurSpan(textNode.textContent);
      textNode.parentNode.replaceChild(span, textNode);
    }
  } catch {
    // backend unavailable — fail silently
  }
}

function collectTextNodes(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const el = node.parentElement;
      if (!el) return NodeFilter.FILTER_REJECT;
      if (el.hasAttribute(PROCESSED)) return NodeFilter.FILTER_REJECT;
      const tag = el.tagName;
      if (['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEXTAREA', 'INPUT'].includes(tag)) {
        return NodeFilter.FILTER_REJECT;
      }
      if (node.textContent.trim().length < 2) return NodeFilter.FILTER_SKIP;
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  const nodes = [];
  let node;
  while ((node = walker.nextNode())) {
    node.parentElement.setAttribute(PROCESSED, 'true');
    nodes.push(node);
  }
  return nodes;
}

function scan(root) {
  const nodes = collectTextNodes(root);
  nodes.forEach(classifyNode);
}

function makeDangerBadge() {
  const badge = document.createElement('span');
  badge.textContent = '⚠ Unsafe Link';
  badge.style.cssText = [
    'background: #ff4444',
    'color: white',
    'font-size: 11px',
    'font-weight: bold',
    'padding: 1px 5px',
    'border-radius: 3px',
    'margin-left: 4px',
    'cursor: default',
    'vertical-align: middle',
  ].join(';');
  badge.title = 'KidShield: this link may be a phishing site';
  return badge;
}

async function classifyLink(anchor) {
  if (anchor.hasAttribute(LINK_PROCESSED)) return;
  anchor.setAttribute(LINK_PROCESSED, 'true');

  const url = anchor.href;
  if (!url || url.startsWith('javascript') || url.startsWith('mailto')) return;

  let hostname;
  try {
    hostname = new URL(url).hostname;
  } catch {
    return;
  }

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: hostname }),
    });
    if (!res.ok) return;
    const { label } = await res.json();
    if (label === 'unsafe') {
      anchor.style.cssText = 'color: #ff4444; text-decoration: line-through;';
      anchor.after(makeDangerBadge());
    }
  } catch {
    // backend unavailable — fail silently
  }
}

function scanLinks(root) {
  const anchors = root.querySelectorAll
    ? root.querySelectorAll(`a[href]:not([${LINK_PROCESSED}])`)
    : [];
  anchors.forEach(classifyLink);
}

async function imageToBase64(img) {
  if (img.src.startsWith('data:')) {
    // already a data URL — strip the header
    const comma = img.src.indexOf(',');
    return comma !== -1 ? img.src.slice(comma + 1) : img.src;
  }
  const res = await fetch(img.src);
  const blob = await res.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const comma = reader.result.indexOf(',');
      resolve(comma !== -1 ? reader.result.slice(comma + 1) : reader.result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function classifyImage(img) {
  if (img.hasAttribute(IMG_PROCESSED)) return;
  img.setAttribute(IMG_PROCESSED, 'true');

  // skip small images
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  if (w < 100 || h < 100) return;

  let imageB64;
  try {
    imageB64 = await imageToBase64(img);
  } catch {
    return; // can't read the image — skip
  }

  try {
    const res = await fetch(IMAGE_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageB64 }),
    });
    if (!res.ok) return;
    const { label } = await res.json();
    if (label === 'unsafe') {
      img.style.cssText = [
        'filter: blur(10px)',
        'cursor: pointer',
        'transition: filter 0.2s',
      ].join(';');
      img.title = 'KidShield: unsafe image hidden — click to reveal';
      img.addEventListener('click', () => {
        img.style.filter = 'none';
        img.style.cursor = 'default';
        img.title = '';
      }, { once: true });
    }
  } catch {
    // backend unavailable — fail silently
  }
}

function scanImages(root) {
  const imgs = root.querySelectorAll
    ? root.querySelectorAll(`img[src]:not([${IMG_PROCESSED}])`)
    : [];
  imgs.forEach(img => {
    if (img.complete && img.naturalWidth > 0) {
      classifyImage(img);
    } else {
      img.addEventListener('load', () => classifyImage(img), { once: true });
    }
  });
}

// first scan
scan(document.body);
scanLinks(document.body);
scanImages(document.body);

// watch for dynamically added content(infinite scroll type of content)
const observer = new MutationObserver((mutations) => {
  for (const mutation of mutations) {
    for (const added of mutation.addedNodes) {
      if (added.nodeType === Node.ELEMENT_NODE && !added.hasAttribute(PROCESSED)) {
        scan(added);
        scanLinks(added);
        scanImages(added);
      }
    }
  }
});

observer.observe(document.body, { childList: true, subtree: true });
