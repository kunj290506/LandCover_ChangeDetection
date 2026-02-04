/**
 * LandCover AI - Production Runtime
 * v2.5.0 "Gold" | Robust Error Handling & Interaction
 */

document.addEventListener('DOMContentLoaded', () => {

    // CONFIGURATION
    const CONF = {
        threshold: 0.5,
        rootMargin: "-10% 0px -10% 0px",
        scrollSmooth: true
    };

    /**
     * 1. STICKY SCROLL INTERACTION
     * Handles the cross-fade between steps/images using IntersectionObserver.
     */
    const scrollSteps = document.querySelectorAll('.text-step');
    const scrollImages = document.querySelectorAll('.sticky-img');

    if (scrollSteps.length > 0 && scrollImages.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Deactivate all steps
                    scrollSteps.forEach(s => s.classList.remove('active'));
                    // Activate current step
                    entry.target.classList.add('active');

                    // Sync Image
                    const idx = entry.target.dataset.index;
                    scrollImages.forEach(img => img.classList.remove('active'));
                    if (scrollImages[idx]) {
                        scrollImages[idx].classList.add('active');
                    }
                }
            });
        }, {
            threshold: CONF.threshold,
            rootMargin: CONF.rootMargin
        });

        scrollSteps.forEach(s => observer.observe(s));
    }


    /**
     * 2. WORKBENCH APPLICATION LOGIC
     * Handles file uploads, preview generation, and API communication.
     */
    const App = {
        inputs: [document.getElementById('input1'), document.getElementById('input2')],
        zones: [document.getElementById('dropZone1'), document.getElementById('dropZone2')],
        previews: [document.getElementById('preview1'), document.getElementById('preview2')],
        analyzeBtn: document.getElementById('analyzeBtn'),
        files: [null, null],

        init() {
            if (!this.zones[0]) return; // Guard clause if app section missing

            this.zones.forEach((zone, i) => {
                // Click to upload
                zone.addEventListener('click', () => this.inputs[i].click());

                // Drag & Drop visual feedback
                zone.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    zone.style.borderColor = 'var(--accent)';
                    zone.style.backgroundColor = '#FFF9F6';
                });
                zone.addEventListener('dragleave', (e) => {
                    e.preventDefault();
                    zone.style.borderColor = ''; // Revert to CSS default
                    zone.style.backgroundColor = '';
                });
                zone.addEventListener('drop', (e) => {
                    e.preventDefault();
                    zone.style.borderColor = '';
                    zone.style.backgroundColor = '';
                    if (e.dataTransfer.files.length) {
                        this.handleFile(e.dataTransfer.files[0], i);
                    }
                });

                // Input Change
                this.inputs[i].addEventListener('change', () => {
                    if (this.inputs[i].files.length) {
                        this.handleFile(this.inputs[i].files[0], i);
                    }
                });
            });

            // Analyze Button
            if (this.analyzeBtn) {
                this.analyzeBtn.addEventListener('click', () => this.runAnalysis());
            }
        },

        handleFile(file, index) {
            // Validation: Image only
            if (!file.type.startsWith('image/')) {
                alert("Please upload a valid image file (PNG, JPG, TIFF).");
                return;
            }

            this.files[index] = file;

            // UI Update
            const zone = this.zones[index];
            const preview = this.previews[index];

            // Hide text, show loading or preview
            zone.querySelector('.upload-content').style.display = 'none';
            if (preview) {
                const reader = new FileReader();
                reader.onload = e => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }

            // Mark active state
            zone.classList.add('active');

            this.checkReady();
        },

        checkReady() {
            if (this.files[0] && this.files[1]) {
                this.analyzeBtn.disabled = false;
                this.analyzeBtn.innerText = "Initialize Sequence";
                this.analyzeBtn.classList.add('ready');
            }
        },

        async runAnalysis() {
            // UI Loading State
            this.analyzeBtn.disabled = true;
            this.analyzeBtn.innerHTML = `<span class="spinner"></span> Processing...`;

            try {
                const formData = new FormData();
                formData.append('image1', this.files[0]);
                formData.append('image2', this.files[1]);

                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Analysis failed');

                const data = await response.json();
                this.renderResults(data);

                this.analyzeBtn.innerText = "Sequence Complete";
                this.analyzeBtn.classList.remove('ready'); // Validated

            } catch (error) {
                console.error("Analysis Error:", error);
                alert("An error occurred during analysis. Please try again.");
                this.analyzeBtn.disabled = false;
                this.analyzeBtn.innerText = "Retry Analysis";
            }
        },

        renderResults(data) {
            const resultsEl = document.getElementById('resultsContainer');
            const emptyEl = document.getElementById('emptyState');

            if (resultsEl && emptyEl) {
                // Fade transition
                emptyEl.style.opacity = '0';
                setTimeout(() => {
                    emptyEl.style.display = 'none';
                    resultsEl.style.display = 'block';

                    // Force reflow for animation
                    void resultsEl.offsetWidth;
                    resultsEl.style.opacity = '1';
                }, 300);

                // Populate Data
                document.getElementById('valTime').innerText = data.process_time + 's';
                document.getElementById('valArea').innerText = new Intl.NumberFormat().format(data.changed_pixels);
                document.getElementById('valPercent').innerText = data.change_percentage.toFixed(2) + '%';

                document.getElementById('resMask').src = data.mask_url;
                document.getElementById('resOverlay').src = data.overlay_url;

                // Scroll to results if needed
                resultsEl.parentElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    };

    // Initialize App
    App.init();

});
