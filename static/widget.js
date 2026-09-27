(function () {
    const script = document.currentScript;
    const baseUrl = script && script.src ? new URL(script.src).origin : window.location.origin;
    const botId = script && script.dataset.botId ? script.dataset.botId : '';
    const title = script && script.dataset.title ? script.dataset.title : 'Qwen AI';

    const root = document.createElement('div');
    root.id = 'botai-widget-root';
    root.style.position = 'fixed';
    root.style.right = '22px';
    root.style.bottom = '22px';
    root.style.zIndex = '2147483647';
    root.style.fontFamily = 'Inter, Arial, sans-serif';

    const frame = document.createElement('iframe');
    const query = botId ? `?bot_id=${encodeURIComponent(botId)}&widget=1` : '?widget=1';
    frame.src = `${baseUrl}/${query}`;
    frame.title = title;
    frame.style.width = '390px';
    frame.style.height = '620px';
    frame.style.maxWidth = 'calc(100vw - 24px)';
    frame.style.maxHeight = 'calc(100vh - 96px)';
    frame.style.border = '1px solid rgba(148, 163, 184, 0.28)';
    frame.style.borderRadius = '18px';
    frame.style.boxShadow = '0 24px 70px rgba(15, 23, 42, 0.38)';
    frame.style.display = 'none';
    frame.style.background = '#0f172a';

    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = title;
    button.style.border = '0';
    button.style.borderRadius = '999px';
    button.style.padding = '13px 18px';
    button.style.background = 'linear-gradient(135deg, #38bdf8, #2563eb)';
    button.style.color = '#fff';
    button.style.fontWeight = '700';
    button.style.cursor = 'pointer';
    button.style.boxShadow = '0 14px 34px rgba(37, 99, 235, 0.36)';

    button.addEventListener('click', () => {
        const isOpen = frame.style.display !== 'none';
        frame.style.display = isOpen ? 'none' : 'block';
        button.textContent = isOpen ? title : 'Đóng chat';
    });

    root.appendChild(frame);
    root.appendChild(button);
    document.addEventListener('DOMContentLoaded', () => document.body.appendChild(root));
})();
