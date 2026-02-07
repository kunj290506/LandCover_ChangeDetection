/**
 * LandCover AI - High-Fidelity Motion System (10/10)
 * Stack: GSAP, Lenis, Vanilla JS
 */

document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // Route Logic
    if (document.querySelector('.landing-page')) {
        initLandingPage();
        initCursor(); // Initialize Cursor only on landing
    } else if (document.querySelector('.app-page')) {
        initAppPage();
    }
});

function initCursor() {
    const cursor = document.querySelector('.cursor');
    const follower = document.querySelector('.cursor-follower');

    if (!cursor || !follower) return;

    let posX = 0, posY = 0;
    let mouseX = 0, mouseY = 0;

    // Smooth Follower
    setInterval(() => {
        posX += (mouseX - posX) * 0.15; // Smooth factor
        posY += (mouseY - posY) * 0.15;

        follower.style.left = posX + 'px';
        follower.style.top = posY + 'px';
        cursor.style.left = mouseX + 'px';
        cursor.style.top = mouseY + 'px';
    }, 10);

    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    // Interaction Scopes
    const links = document.querySelectorAll('a, button, .bento-item, .nav-item');
    links.forEach(link => {
        link.addEventListener('mouseenter', () => {
            cursor.classList.add('link');
            follower.classList.add('link');
        });
        link.addEventListener('mouseleave', () => {
            cursor.classList.remove('link');
            follower.classList.remove('link');
        });
    });
}

/* ===================================
   LANDING PAGE (Cinematic)
   =================================== */

function initLandingPage() {
    // 1. Initialize Lenis (Premium Inertia)
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

    // 2. Navbar Scroll Effect (Dynamic Island shrink)
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // 3. GSAP Orchestration
    if (typeof gsap !== 'undefined') {
        gsap.registerPlugin(ScrollTrigger);
        runHeroSequence();
        runBentoSequence();
    }
}

function runHeroSequence() {
    const tl = gsap.timeline();

    tl.from(".hero-title", {
        y: 30,
        opacity: 0,
        duration: 0.8,
        ease: "power4.out"
    })
        .from(".hero-subtitle", {
            y: 15,
            opacity: 0,
            duration: 0.6
        }, "-=0.5")
        .from(".hero-actions", {
            y: 15,
            opacity: 0,
            duration: 0.6
        }, "-=0.4");
}

function runBentoSequence() {
    // Stagger Fade In for Bento Cards
    const items = gsap.utils.toArray('.bento-item');
    if (items.length === 0) return;

    items.forEach((item, i) => {
        gsap.from(item, {
            scrollTrigger: {
                trigger: item,
                start: "top 90%", // Trigger earlier
                toggleActions: "play none none reverse"
            },
            y: 50,
            opacity: 0,
            duration: 1,
            ease: "power3.out",
            delay: i * 0.1
        });
    });
}

/* ===================================
   APP PAGE (Native Mac Feel)
   =================================== */

function initAppPage() {
    const AppState = {
        files: [null, null],
        isProcessing: false
    };

    const elements = {
        dropZones: [document.getElementById('dropZone1'), document.getElementById('dropZone2')],
        inputs: [document.getElementById('input1'), document.getElementById('input2')],
        placeholders: [document.getElementById('placeholder1'), document.getElementById('placeholder2')],

        // Preview Wrappers
        previewContainers: [document.getElementById('previewContainer1'), document.getElementById('previewContainer2')],
        previewImages: [document.getElementById('preview1'), document.getElementById('preview2')],

        analyzeBtn: document.getElementById('analyzeBtn'),
        loadingState: document.getElementById('loadingState'),
        emptyState: document.getElementById('emptyState'),
        resultsDisplay: document.getElementById('resultsDisplay'),

        // Metrics
        miniMetrics: document.getElementById('miniMetrics'),
        metricChange: document.getElementById('metricChange'),
        metricTime: document.getElementById('metricTime')
    };

    // Initialize Interactions
    elements.dropZones.forEach((zone, index) => {
        if (!zone) return;

        // Click to Upload
        zone.addEventListener('click', (e) => {
            // If checking for remove button, do it here or let bubble?
            // Sidebar structure: .drop-zone contains .preview-wrapper.
            // But usually we hide the drop zone specific contents.
            // Let's rely on the input click.
            if (!AppState.files[index]) {
                elements.inputs[index].click();
            }
        });

        // Input Change
        elements.inputs[index].addEventListener('change', (e) => {
            if (e.target.files.length) handleFile(e.target.files[0], index);
        });

        // Drag & Drop
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            if (!AppState.files[index]) {
                zone.style.borderColor = 'var(--accent)';
                zone.style.background = '#F0F8FF';
            }
        });

        zone.addEventListener('dragleave', () => {
            if (!AppState.files[index]) {
                zone.style.borderColor = 'rgba(0,0,0,0.1)';
                zone.style.background = 'rgba(255,255,255,0.5)';
            }
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.style.borderColor = 'rgba(0,0,0,0.1)';
            zone.style.background = 'rgba(255,255,255,0.5)';
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0], index);
        });
    });

    function handleFile(file, index) {
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file.');
            return;
        }

        AppState.files[index] = file;

        // Read & Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            // Update Preview Image
            elements.previewImages[index].src = e.target.result;

            // Layout Shift: Hide Placeholder, Show Preview
            elements.placeholders[index].style.display = 'none';
            elements.previewContainers[index].style.display = 'flex';

            // Add 'has-file' class to dropzone for styling
            elements.dropZones[index].classList.add('has-file');

            checkReady();
        };
        reader.readAsDataURL(file);
    }

    // Remove Logic (Delegate)
    document.querySelectorAll('.p-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation(); // Stop bubble to dropzone
            const index = parseInt(btn.dataset.index) - 1;

            AppState.files[index] = null;
            elements.inputs[index].value = '';

            // Reset UI
            elements.previewContainers[index].style.display = 'none';
            elements.placeholders[index].style.display = 'block';
            elements.dropZones[index].classList.remove('has-file');
            elements.dropZones[index].style.background = 'rgba(255,255,255,0.5)'; // Reset bg

            checkReady();
        });
    });

    function checkReady() {
        const ready = AppState.files[0] && AppState.files[1];
        elements.analyzeBtn.disabled = !ready || AppState.isProcessing;
    }

    // Analyze Action
    elements.analyzeBtn.addEventListener('click', async () => {
        if (elements.analyzeBtn.disabled) return;

        AppState.isProcessing = true;
        checkReady();

        // UI Transition
        elements.emptyState.style.display = 'none';
        elements.resultsDisplay.style.display = 'none'; // Reset result view
        elements.loadingState.style.display = 'flex';

        // Prepare Payload
        const formData = new FormData();
        formData.append('image1', AppState.files[0]);
        formData.append('image2', AppState.files[1]);

        try {
            const res = await fetch('/api/predict', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.error) throw new Error(data.error);

            // Success
            showResults(data);

        } catch (err) {
            console.error(err);
            alert("Analysis Error: " + err.message);
            elements.emptyState.style.display = 'block'; // Revert
        } finally {
            AppState.isProcessing = false;
            elements.loadingState.style.display = 'none';
            checkReady();
        }
    });

    function showResults(data) {
        elements.resultsDisplay.style.display = 'block';

        // Metrics
        elements.miniMetrics.style.display = 'block';
        elements.metricChange.textContent = data.change_percentage.toFixed(2) + '%';
        elements.metricTime.textContent = (data.process_time * 1000).toFixed(0) + 'ms';

        // Images
        document.getElementById('compBefore').src = data.image1_url;
        document.getElementById('compAfter').src = data.image2_url;

        // Inspector References
        document.getElementById('refBefore').src = data.image1_url;
        document.getElementById('refAfter').src = data.image2_url;

        document.getElementById('resultMask').src = data.mask_url;

        initSlider();

        // Slide Up Animation
        gsap.fromTo(elements.resultsDisplay,
            { y: 20, opacity: 0 },
            { y: 0, opacity: 1, duration: 0.6, ease: "power2.out" }
        );
    }

    function initSlider() {
        const container = document.querySelector('.compare-container');
        const handle = document.querySelector('.comp-handle'); // The line
        const beforeLayer = document.querySelector('.comparison-before'); // The top layer

        // In this setup:
        // 'comparison-before' is TOP layer. 'comparison-after' is BOTTOM.
        // As handle moves right (val 0->100), we reveal more of the Left side (Before)?
        // Wait, standard sliders: Left side = Img A, Right side = Img B.
        // If handle is at 50%, left 50% is A, right 50% is B.
        // clip-path: inset(0 X% 0 0) clips from the RIGHT side.
        // So applied to top layer (Before), inset(0 50% 0 0) means right 50% is cut, showing bottom (After).
        // Correct.

        let active = false;

        const setPosition = (x) => {
            const rect = container.getBoundingClientRect();
            let p = (x - rect.left) / rect.width;
            p = Math.max(0, Math.min(1, p));

            const pct = p * 100;

            handle.style.left = pct + '%';
            beforeLayer.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
        };

        handle.addEventListener('mousedown', () => active = true);
        window.addEventListener('mouseup', () => active = false);
        window.addEventListener('mousemove', (e) => {
            if (active) setPosition(e.clientX);
        });

        // Initial set
        setPosition(container.getBoundingClientRect().left + (container.offsetWidth * 0.5));
    }
}
