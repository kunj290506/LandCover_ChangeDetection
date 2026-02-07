/**
 * LandCover AI - Premium Interactive Experience
 * Stack: GSAP, Lenis, Vanilla JS
 */

document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // Global Components
    initTheme();
    initCursor();

    // Route Logic
    if (document.querySelector('.landing-page')) {
        initLandingPage();
    } else if (document.querySelector('.app-page')) {
        initAppPage();
    }
});

/* ═══════════════════════════════════════════════════════════════════════════
   GLOBAL COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */

function initTheme() {
    const savedTheme = localStorage.getItem('landcover-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialTheme = savedTheme || (prefersDark ? 'dark' : 'light');

    document.documentElement.setAttribute('data-theme', initialTheme);

    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('landcover-theme', next);
        });
    }
}

function initCursor() {
    const cursorDot = document.getElementById('cursorDot');
    const cursorOutline = document.getElementById('cursorOutline');

    if (!cursorDot || !cursorOutline || 'ontouchstart' in window) {
        if (cursorDot) cursorDot.style.display = 'none';
        if (cursorOutline) cursorOutline.style.display = 'none';
        return;
    }

    // Mouse flow
    window.addEventListener('mousemove', (e) => {
        const posX = e.clientX;
        const posY = e.clientY;

        // Dot follows instantly
        cursorDot.style.transform = `translate(${posX}px, ${posY}px) translate(-50%, -50%)`;

        // Outline follows with lag (handled by CSS transition or simple JS loop)
        // For smoother feel, we can use requestAnimationFrame or GSAP quickSetter
        // Using simple GSAP for consistency if available, else direct
        if (typeof gsap !== 'undefined') {
            gsap.to(cursorOutline, {
                x: posX,
                y: posY,
                duration: 0.15,
                ease: "power2.out"
            });
        }
    });

    // Hover interactions
    const hoverTargets = document.querySelectorAll('a, button, .feature-card, .upload-zone');
    hoverTargets.forEach(el => {
        el.addEventListener('mouseenter', () => document.body.classList.add('hovering'));
        el.addEventListener('mouseleave', () => document.body.classList.remove('hovering'));
    });
}

/* ═══════════════════════════════════════════════════════════════════════════
   LANDING PAGE LOGIC (GSAP + Lenis)
   ═══════════════════════════════════════════════════════════════════════════ */

function initLandingPage() {
    // 1. Initialize Lenis Smooth Scroll
    const lenis = new Lenis({
        duration: 1.2,
        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        smooth: true,
        direction: 'vertical'
    });

    function raf(time) {
        lenis.raf(time);
        requestAnimationFrame(raf);
    }
    requestAnimationFrame(raf);

    // Integrate ScrollTrigger with Lenis
    if (typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {
        gsap.registerPlugin(ScrollTrigger);
        // SplitType needs to be loaded
    }

    // 2. Hero Animations
    initHeroAnimations();

    // 3. Section Animations
    initScrollAnimations();

    // 4. Mobile Menu
    const mobileToggle = document.getElementById('mobileToggle');
    const navLinks = document.querySelector('.nav-links');
    if (mobileToggle) {
        mobileToggle.addEventListener('click', () => {
            mobileToggle.classList.toggle('active');
            navLinks.classList.toggle('active');
        });
    }

    // Navbar Scroll Effect
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) navbar.classList.add('scrolled');
        else navbar.classList.remove('scrolled');
    });
}

function initHeroAnimations() {
    if (typeof gsap === 'undefined') return;

    const tl = gsap.timeline({ defaults: { ease: "power3.out" } });

    // Split text logic would go here if we had the library loaded locally, 
    // but assuming simple opacity/y translates for now

    tl.from(".hero-label", {
        y: 20,
        opacity: 0,
        duration: 1,
        delay: 0.2
    })
        .from(".title-line", {
            y: 100,
            opacity: 0,
            duration: 1.2,
            stagger: 0.15,
            ease: "power4.out"
        }, "-=0.8")
        .from(".hero-description", {
            y: 20,
            opacity: 0,
            duration: 1
        }, "-=0.6")
        .from(".hero-actions > *", {
            y: 20,
            opacity: 0,
            stagger: 0.1,
            duration: 1
        }, "-=0.8")
        .from(".stat-item", {
            y: 20,
            opacity: 0,
            stagger: 0.1,
            duration: 1
        }, "-=0.8");

    // Animate stats numbers
    const stats = document.querySelectorAll('.stat-value');
    stats.forEach(stat => {
        const val = parseFloat(stat.dataset.value);
        gsap.to(stat, {
            innerText: val,
            duration: 2,
            snap: { innerText: 0.1 },
            ease: "power1.inOut"
        });
    });
}

function initScrollAnimations() {
    if (typeof gsap === 'undefined') return;

    // Features Stagger
    gsap.from(".feature-card", {
        scrollTrigger: {
            trigger: ".features-grid",
            start: "top 80%",
        },
        y: 50,
        opacity: 0,
        duration: 0.8,
        stagger: 0.1
    });

    // Tech Stack
    const techItems = document.querySelectorAll('.tech-item');
    techItems.forEach((item, index) => {
        gsap.from(item, {
            scrollTrigger: {
                trigger: item,
                start: "top 85%",
            },
            x: -30,
            opacity: 0,
            duration: 0.6,
            delay: index * 0.1
        });
    });
}

/* ═══════════════════════════════════════════════════════════════════════════
   APP PAGE LOGIC (Change Detection)
   ═══════════════════════════════════════════════════════════════════════════ */

function initAppPage() {
    const AppState = {
        files: [null, null],
        isProcessing: false,
        results: null
    };

    const elements = {
        dropZones: [document.getElementById('dropZone1'), document.getElementById('dropZone2')],
        inputs: [document.getElementById('input1'), document.getElementById('input2')],
        previews: [document.getElementById('preview1'), document.getElementById('preview2')],
        previewContainers: [document.getElementById('previewContainer1'), document.getElementById('previewContainer2')],
        placeholders: [document.getElementById('placeholder1'), document.getElementById('placeholder2')],
        analyzeBtn: document.getElementById('analyzeBtn'),
        clearBtn: document.getElementById('clearBtn'),
        emptyState: document.getElementById('emptyState'),
        loadingState: document.getElementById('loadingState'),
        resultsDisplay: document.getElementById('resultsDisplay'),
        statusIndicator: document.getElementById('statusIndicator'),
        toastContainer: document.getElementById('toastContainer')
    };

    // Initialize Drop Zones
    elements.dropZones.forEach((zone, index) => {
        if (!zone) return;

        zone.addEventListener('click', (e) => {
            if (!e.target.closest('.remove-btn-simple')) elements.inputs[index].click();
        });

        elements.inputs[index].addEventListener('change', (e) => {
            if (e.target.files.length) handleFile(e.target.files[0], index);
        });

        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });

        zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0], index);
        });
    });

    function handleFile(file, index) {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            showToast('error', 'Invalid File', 'Please upload PNG, JPG, or TIFF.');
            return;
        }

        AppState.files[index] = file;

        // Update Metadata
        const sizeStr = formatBytes(file.size);
        const typeStr = file.type.split('/')[1].toUpperCase();

        const nameEl = document.getElementById(`fileName${index + 1}`);
        const metaEl = document.getElementById(`fileMeta${index + 1}`);

        if (nameEl) nameEl.textContent = file.name;
        if (metaEl) metaEl.textContent = `${sizeStr} • ${typeStr}`;

        const reader = new FileReader();
        reader.onload = (e) => {
            elements.previews[index].src = e.target.result;
            elements.placeholders[index].style.display = 'none';
            elements.previewContainers[index].style.display = 'flex'; // Changed to flex for new layout
            updateButtons();
        };
        reader.readAsDataURL(file);
    }

    function formatBytes(bytes, decimals = 1) {
        if (!+bytes) return '0 B';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }

    // Remove Image
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('.remove-btn-simple');
        if (!btn) return;

        e.stopPropagation();
        const index = parseInt(btn.dataset.index) - 1;
        AppState.files[index] = null;
        elements.inputs[index].value = '';
        elements.previews[index].src = '';
        elements.placeholders[index].style.display = 'flex'; // Revert to flex
        elements.previewContainers[index].style.display = 'none';
        updateButtons();
    });

    function updateButtons() {
        const hasFiles = AppState.files[0] && AppState.files[1];
        if (elements.analyzeBtn) elements.analyzeBtn.disabled = !hasFiles || AppState.isProcessing;
    }

    if (elements.clearBtn) {
        elements.clearBtn.addEventListener('click', () => {
            // Reset logic
            location.reload();
        });
    }

    // Analysis Logic
    if (elements.analyzeBtn) {
        elements.analyzeBtn.addEventListener('click', async () => {
            if (!AppState.files[0] || !AppState.files[1]) return;

            AppState.isProcessing = true;
            updateButtons();

            elements.emptyState.style.display = 'none';
            elements.loadingState.style.display = 'block';
            if (elements.resultsDisplay) elements.resultsDisplay.style.display = 'none';

            const formData = new FormData();
            formData.append('image1', AppState.files[0]);
            formData.append('image2', AppState.files[1]);

            try {
                // Mock progress
                let progress = 0;
                const progressBar = document.getElementById('progressBar');
                const interval = setInterval(() => {
                    progress = Math.min(progress + 5, 90);
                    if (progressBar) progressBar.style.width = `${progress}%`;
                }, 100);

                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });

                clearInterval(interval);
                if (progressBar) progressBar.style.width = '100%';

                const data = await response.json();

                if (data.error) throw new Error(data.error);

                // Show Results
                elements.loadingState.style.display = 'none';
                elements.resultsDisplay.style.display = 'flex'; // Important: flex for layout
                elements.resultsDisplay.classList.add('active');

                displayResults(data);
                showToast('success', 'Analysis Complete', `Finished in ${data.process_time}s`);

            } catch (error) {
                console.error(error);
                elements.loadingState.style.display = 'none';
                elements.emptyState.style.display = 'block';
                showToast('error', 'Analysis Failed', error.message);
            } finally {
                AppState.isProcessing = false;
                updateButtons();
            }
        });
    }

    function displayResults(data) {
        // Images
        if (document.getElementById('compBefore')) document.getElementById('compBefore').src = data.image1_url;
        if (document.getElementById('compAfter')) document.getElementById('compAfter').src = data.image2_url;
        if (document.getElementById('resultMask')) document.getElementById('resultMask').src = data.mask_url;
        if (document.getElementById('resultOverlay')) document.getElementById('resultOverlay').src = data.overlay_url;

        // Metrics
        document.getElementById('metricChange').textContent = `${data.change_percentage.toFixed(2)}%`;
        document.getElementById('metricPixels').textContent = data.changed_pixels.toLocaleString();
        document.getElementById('metricTotal').textContent = data.total_pixels.toLocaleString();
        document.getElementById('metricConfidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
        document.getElementById('metricTime').textContent = `${(data.process_time * 1000).toFixed(0)}ms`;

        // Initialize Slider
        initComparisonSlider();

        // Setup Download Buttons
        document.getElementById('downloadMask').onclick = () => download(data.mask_url, 'mask.png');
        document.getElementById('downloadOverlay').onclick = () => download(data.overlay_url, 'overlay.png');
    }

    // Comparison Slider Logic
    function initComparisonSlider() {
        const slider = document.getElementById('comparisonSlider');
        const handle = document.getElementById('comparisonHandle');
        const beforeImg = document.querySelector('.comparison-before');

        if (!slider || !handle || !beforeImg) return;

        let active = false;

        const update = (x) => {
            const rect = slider.getBoundingClientRect();
            let val = ((x - rect.left) / rect.width) * 100;
            val = Math.max(0, Math.min(100, val));

            handle.style.left = `${val}%`;
            beforeImg.style.clipPath = `inset(0 ${100 - val}% 0 0)`;
        };

        const start = () => active = true;
        const end = () => active = false;
        const move = (e) => {
            if (!active) return;
            const x = e.touches ? e.touches[0].clientX : e.clientX;
            update(x);
        };

        handle.addEventListener('mousedown', start);
        window.addEventListener('mouseup', end);
        window.addEventListener('mousemove', move);

        handle.addEventListener('touchstart', start);
        window.addEventListener('touchend', end);
        window.addEventListener('touchmove', move);
    }

    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.result-tab-content').forEach(c => {
                c.classList.remove('active');
                c.style.display = 'none'; // Force hide
            });

            btn.classList.add('active');
            const content = document.getElementById(`tab-${btn.dataset.tab}`);
            if (content) {
                content.classList.add('active');
                content.style.display = 'flex'; // Force show
            }
        });
    });

    // Helper: Toast
    function showToast(type, title, msg) {
        const container = elements.toastContainer;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <strong>${title}</strong>
                <div>${msg}</div>
            </div>
        `;
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }

    // Helper: Download
    function download(url, name) {
        const a = document.createElement('a');
        a.href = url;
        a.download = name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
}
