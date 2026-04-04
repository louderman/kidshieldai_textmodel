const API_URL = 'http://127.0.0.1:8000/classify';
const PROCESSED = 'data-kidshield-processed';

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
  if (!text || text.length < 10) return;

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
      if (node.textContent.trim().length < 10) return NodeFilter.FILTER_SKIP;
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
  if (anchor.hasAttribute(PROCESSED)) return;
  anchor.setAttribute(PROCESSED, 'true');

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
    ? root.querySelectorAll(`a[href]:not([${PROCESSED}])`)
    : [];
  anchors.forEach(classifyLink);
}

// first scan
scan(document.body);
scanLinks(document.body);

// watch for dynamically added content(infinite scroll type of content)
const observer = new MutationObserver((mutations) => {
  for (const mutation of mutations) {
    for (const added of mutation.addedNodes) {
      if (added.nodeType === Node.ELEMENT_NODE && !added.hasAttribute(PROCESSED)) {
        scan(added);
        scanLinks(added);
      }
    }
  }
});

observer.observe(document.body, { childList: true, subtree: true });
