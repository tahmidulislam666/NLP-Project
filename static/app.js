const message = document.querySelector('#message');
const send = document.querySelector('#send');
const chat = document.querySelector('#chat');
const statusTitle = document.querySelector('#status-title');
const statusCopy = document.querySelector('#status-copy');
const violationEl = document.querySelector('#violations');
let timer;

function escapeHTML(value) { const div = document.createElement('div'); div.textContent = value; return div.innerHTML; }
function addBubble(html, type) { const item = document.createElement('article'); item.className = `bubble ${type}`; item.innerHTML = html; chat.append(item); chat.scrollTop = chat.scrollHeight; }
function showCooldown(seconds) {
  document.body.classList.add('blocked'); send.disabled = true; message.disabled = true;
  clearInterval(timer); let remaining = seconds;
  const tick = () => { statusTitle.textContent = `Sending paused · ${remaining}s`; statusCopy.textContent = 'Take a breath, then rejoin the conversation respectfully.'; if (remaining-- <= 0) { clearInterval(timer); document.body.classList.remove('blocked'); send.disabled = false; message.disabled = false; statusTitle.textContent = 'Safety check active'; statusCopy.textContent = 'You can send messages again.'; }};
  tick(); timer = setInterval(tick, 1000);
}
async function submit() {
  const text = message.value.trim(); if (!text) return;
  addBubble(escapeHTML(text), 'user'); message.value = ''; send.disabled = true;
  try {
    const response = await fetch('/api/messages', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:text})});
    const data = await response.json(); violationEl.textContent = data.violations ?? 0;
    if (data.blocked) { addBubble('<strong>Message paused</strong>Your cooldown is still active.', 'alert'); showCooldown(data.seconds_left); return; }
    if (data.unsafe) {
      addBubble(`<strong>Blocked · ${data.severity.toUpperCase()}</strong>${data.category}. Let’s keep the conversation safe.<div class="suggestion">Try instead: “${escapeHTML(data.safer_alternative)}”</div>`, 'alert');
      if (data.cooldown) showCooldown(data.cooldown); else { statusTitle.textContent = 'Warning issued'; statusCopy.textContent = `${data.violations} violation${data.violations === 1 ? '' : 's'} recorded.`; }
    } else { addBubble('Message delivered. Thanks for keeping it constructive.', 'allowed'); statusTitle.textContent = 'Safety check passed'; statusCopy.textContent = 'Your message was classified as safe.'; }
  } catch { addBubble('<strong>Connection problem</strong>Please make sure the moderation server is running.', 'alert'); }
  finally { if (!document.body.classList.contains('blocked')) send.disabled = false; message.focus(); }
}
send.addEventListener('click', submit); message.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }});
document.querySelector('#reset').addEventListener('click', async () => { await fetch('/api/reset', {method:'POST'}); violationEl.textContent = '0'; statusTitle.textContent = 'Safety check active'; statusCopy.textContent = 'English, বাংলা, and code-mixed messages are supported.'; });
