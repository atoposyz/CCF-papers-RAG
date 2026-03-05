document.addEventListener('DOMContentLoaded', () => {
    // ---- DOM Elements ----
    const searchForm = document.getElementById('search-form');
    const searchBtn = document.getElementById('search-btn');
    const searchSpinner = document.getElementById('search-spinner');
    const resultsContainer = document.getElementById('results-container');

    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const sendBtn = document.getElementById('send-btn');
    const clearChatBtn = document.getElementById('clear-chat-btn');

    const apiKeyInput = document.getElementById('api-key');
    const apiBaseInput = document.getElementById('api-base');
    const modelNameInput = document.getElementById('model-name');
    const toast = document.getElementById('toast');

    // ---- Weight Slider ----
    const vectorWeightSlider = document.getElementById('vector-weight');
    const weightHint = document.getElementById('weight-hint');

    const updateSliderStyle = () => {
        const vw = parseInt(vectorWeightSlider.value, 10);
        const bw = 100 - vw;
        weightHint.textContent = `Vector ${vw}% · BM25 ${bw}%`;
        // 更新 CSS 变量，驱动背景颜色渐变
        vectorWeightSlider.style.setProperty('--val', vw);
    };

    vectorWeightSlider.addEventListener('input', updateSliderStyle);
    // 初始化一次
    updateSliderStyle();

    // ---- Collapsible Settings ----
    const toggleSettingsBtn = document.getElementById('toggle-settings');
    const settingsContainer = document.getElementById('settings-container');

    toggleSettingsBtn.addEventListener('click', () => {
        const isCollapsed = settingsContainer.style.display === 'none';
        settingsContainer.style.display = isCollapsed ? 'block' : 'none';
        toggleSettingsBtn.querySelector('.toggle-icon').style.transform = isCollapsed ? 'rotate(0deg)' : 'rotate(180deg)';
    });

    // ---- State ----
    let currentContextPapers = "";
    let messageHistory = [];

    // ---- Utilities ----
    function showToast(message, type = 'error') {
        toast.textContent = message;
        toast.className = `toast show ${type}`;
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    // ---- Search Logic ----
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const query = document.getElementById('query-input').value.trim();
        const venueRaw = document.getElementById('venue-input').value.trim();
        const year = document.getElementById('year-input').value.trim();
        let topK = parseInt(document.getElementById('topk-input').value.trim(), 10);
        if (isNaN(topK) || topK < 1) topK = 5;

        // 解析多选 Venue
        const venue = venueRaw ? venueRaw.split(',').map(v => v.trim()).filter(v => v) : null;

        const vwRaw = parseInt(vectorWeightSlider.value, 10);
        const vectorWeight = vwRaw / 100;
        const bm25Weight = (100 - vwRaw) / 100;

        if (!query) return;

        // UI Loading State
        searchBtn.disabled = true;
        searchSpinner.style.display = 'block';
        searchBtn.querySelector('span').style.opacity = '0';
        resultsContainer.innerHTML = '';

        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    venue,
                    year,
                    top_k: topK,
                    vector_weight: vectorWeight,
                    bm25_weight: bm25Weight
                })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            renderSearchResults(data.results);

            // Build Context for RAG
            buildRagContext(data.results, query);

        } catch (error) {
            console.error('Search failed:', error);
            showToast('Search request failed. Backend might be down.');
            resultsContainer.innerHTML = `<div class="placeholder-text text-error">Failed to load results.</div>`;
        } finally {
            searchBtn.disabled = false;
            searchSpinner.style.display = 'none';
            searchBtn.querySelector('span').style.opacity = '1';
        }
    });

    function renderSearchResults(results) {
        if (!results || results.length === 0) {
            resultsContainer.innerHTML = `<div class="placeholder-text">No papers found matching your criteria.</div>`;
            return;
        }

        results.forEach(paper => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.dataset.id = paper.id;

            // Build Link HTML
            let linksHtml = '';
            if (paper.doi_url) {
                linksHtml += `<a href="${paper.doi_url}" target="_blank" class="card-link doi-link" title="Open DOI">📄 DOI</a>`;
            }
            if (paper.dblp_url) {
                linksHtml += `<a href="${paper.dblp_url}" target="_blank" class="card-link dblp-link" title="Open DBLP">📚 DBLP</a>`;
            }

            // Score badges
            const hybridBadge = `<span class="meta-badge score-hybrid" title="Hybrid RRF Score (higher is better)">⚡ Score: ${paper.hybrid_score}</span>`;
            const vectorBadge = paper.vector_dist !== null
                ? `<span class="meta-badge score-vector" title="Vector semantic distance (lower is more similar)">🧠 VecDist: ${paper.vector_dist}</span>`
                : `<span class="meta-badge score-vector muted" title="Not in vector top results">🧠 Vec: —</span>`;
            const bm25Badge = paper.bm25_score > 0
                ? `<span class="meta-badge score-bm25" title="BM25 keyword score (higher is better)">🔤 BM25: ${paper.bm25_score}</span>`
                : `<span class="meta-badge score-bm25 muted" title="No keyword match">🔤 BM25: —</span>`;

            card.innerHTML = `
                <div class="result-header">
                    <h3 class="result-title">${paper.title}</h3>
                    <div class="expand-icon">
                        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="chevron"><polyline points="6 9 12 15 18 9"></polyline></svg>
                    </div>
                </div>
                <div class="result-tags">
                    <span class="meta-badge venue">${paper.venue}</span>
                    <span class="meta-badge">${paper.year}</span>
                    <span class="meta-badge">${paper.author} (1st Author)</span>
                    ${hybridBadge}
                    ${vectorBadge}
                    ${bm25Badge}
                </div>
                <div class="result-details" style="display: none;">
                    <p class="result-abstract">${paper.abstract}</p>
                    <div class="result-links">
                        ${linksHtml}
                    </div>
                </div>
            `;

            // Toggle Logic
            const header = card.querySelector('.result-header');
            header.style.cursor = 'pointer';
            header.addEventListener('click', () => {
                const details = card.querySelector('.result-details');
                const chevron = card.querySelector('.chevron');
                if (details.style.display === 'none') {
                    details.style.display = 'block';
                    chevron.style.transform = 'rotate(180deg)';
                } else {
                    details.style.display = 'none';
                    chevron.style.transform = 'rotate(0deg)';
                }
            });

            // Add slight fade in animation delay
            card.style.animation = `fadeIn 0.5s ease forwards ${paper.id * 0.1}s`;
            card.style.opacity = '0';
            resultsContainer.appendChild(card);
        });
    }

    function buildRagContext(results, query) {
        if (!results || results.length === 0) {
            currentContextPapers = "未检索到相关的论文上下文。请根据你的自身知识回答，但必须说明本系统数据库内没有相关资料。";
            return;
        }

        // Construct detailed context string
        let context = `用户刚才检索了: "${query}"\n\n`;
        context += "以下是从数据库中检索到的最相关的论文：\n";
        results.forEach((p, index) => {
            context += `\n[${index + 1}] 标题: ${p.title}\n年份: ${p.year}\n会议/期刊: ${p.venue}\n第一作者: ${p.author}\n摘要片段: ${p.abstract}\n`;
        });
        currentContextPapers = context;

        // Inject a system notification to chat
        const infoHtml = `<div class="message ai-message"><div class="message-content" style="background: rgba(16, 185, 129, 0.2); border-color: rgba(16,185,129,0.3)">
            ✅ Retrieved ${results.length} papers for context. You can now ask me questions about them!
        </div></div>`;
        chatMessages.insertAdjacentHTML('beforeend', infoHtml);
        scrollToBottom();
    }

    // ---- Chat Logic ----
    clearChatBtn.addEventListener('click', () => {
        messageHistory = [];
        chatMessages.innerHTML = `
            <div class="message ai-message">
                <div class="message-content">
                    Conversation cleared. How can I help you today?
                </div>
            </div>
        `;
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const userInput = chatInput.value.trim();
        if (!userInput) return;

        const apiKey = apiKeyInput.value.trim();
        if (!apiKey) {
            showToast('Please enter your API Key in the top right corner first.');
            apiKeyInput.focus();
            return;
        }

        // 1. Add User Message to UI and History
        appendMessage('user', userInput);
        messageHistory.push({ role: "user", content: userInput });
        chatInput.value = '';

        // Disable UI
        sendBtn.disabled = true;
        chatInput.disabled = true;

        // 2. Prepare AI Message Block for Streaming
        const aiMessageDiv = document.createElement('div');
        aiMessageDiv.className = 'message ai-message';
        const aiContentDiv = document.createElement('div');
        aiContentDiv.className = 'message-content typing-cursor';
        aiMessageDiv.appendChild(aiContentDiv);
        chatMessages.appendChild(aiMessageDiv);
        scrollToBottom();

        let fullAiResponse = "";

        // 3. Setup SSE Request
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: messageHistory,
                    context_papers: currentContextPapers || "没有提供参考论文上下文。",
                    api_key: apiKey,
                    base_url: apiBaseInput.value.trim(),
                    model: modelNameInput.value.trim()
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP Error ${response.status}`);
            }

            // Stream Reader
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        if (dataStr === '[DONE]') {
                            break;
                        }

                        try {
                            const data = JSON.parse(dataStr);
                            if (data.error) {
                                fullAiResponse += `\n**[Error]**: ${data.error}`;
                            } else if (data.content) {
                                fullAiResponse += data.content;
                                // Render markdown on the fly
                                aiContentDiv.innerHTML = marked.parse(fullAiResponse);
                                scrollToBottom();
                            }
                        } catch (e) {
                            console.error('JSON parse error on streaming chunk:', e);
                        }
                    }
                }
            }

            // Streaming finished
            aiContentDiv.classList.remove('typing-cursor');
            messageHistory.push({ role: "assistant", content: fullAiResponse });

        } catch (error) {
            console.error('Chat error:', error);
            aiContentDiv.classList.remove('typing-cursor');
            aiContentDiv.innerHTML += `<p style="color: #ef4444;">Error: ${error.message}</p>`;
        } finally {
            // Re-enable UI
            sendBtn.disabled = false;
            chatInput.disabled = false;
            chatInput.focus();
        }
    });

    function appendMessage(role, text) {
        const div = document.createElement('div');
        div.className = `message ${role}-message`;

        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = text;

        div.appendChild(content);
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Auto-resize textarea
    chatInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') {
            this.style.height = 'auto'; // Reset
        }
    });
});
