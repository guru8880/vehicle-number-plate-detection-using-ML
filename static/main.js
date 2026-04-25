document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loadingState = document.getElementById('loading');
    const uploadSection = document.querySelector('.upload-section');
    const resultsSection = document.getElementById('results-section');
    const annotatedImage = document.getElementById('annotated-image');
    const platesContainer = document.getElementById('plates-container');
    const resetBtn = document.getElementById('reset-btn');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(event) {
        event.preventDefault();
        event.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach((eventName) => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach((eventName) => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function () {
        if (this.files && this.files.length > 0) {
            handleFiles(this.files);
        }
    });

    function handleDrop(event) {
        const files = event.dataTransfer.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        const file = files[0];
        if (!file || !file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }
        uploadImage(file);
    }

    async function uploadImage(file) {
        uploadSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loadingState.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            loadingState.classList.add('hidden');

            if (!response.ok) {
                throw new Error(data.error || 'Server error occurred');
            }

            displayResults(data);
        } catch (error) {
            alert('Error processing image: ' + error.message);
            loadingState.classList.add('hidden');
            uploadSection.classList.remove('hidden');
        }
    }

    function displayResults(data) {
        annotatedImage.src = data.annotated_image;
        platesContainer.innerHTML = '';

        if (data.plates && data.plates.length > 0) {
            data.plates.forEach((plate) => {
                const timestamp = new Date().toLocaleString();
                const overallPercent = formatPercent(plate.confidence);
                const detectorPercent = formatPercent(plate.detector_confidence);
                const ocrPercent = formatPercent(plate.ocr_confidence);
                const checksumText = plate.is_singapore_plate
                    ? (plate.checksum_match ? 'Passed' : 'Failed')
                    : 'Not applicable';
                let methodText = plate.method === 'best_pt' ? 'Custom detector + OCR' : 'Fallback ALPR';
                if (plate.method.endsWith('_checksum')) {
                    methodText = 'Custom detector + OCR + checksum correction';
                }
                const noteHtml = plate.correction_note
                    ? `
                            <div class="meta-row">
                                <strong>Review Note:</strong>
                                <span>${escapeHtml(plate.correction_note)}</span>
                            </div>
                      `
                    : '';
                const displayNoteHtml = plate.display_note
                    ? `
                            <div class="meta-row">
                                <strong>Display Note:</strong>
                                <span>${escapeHtml(plate.display_note)}</span>
                            </div>
                      `
                    : '';

                const plateHtml = `
                    <div class="plate-container">
                        <div class="real-plate">
                            <span class="plate-text">${escapeHtml(plate.display_plate || plate.plate)}</span>
                        </div>

                        <div class="meta-details">
                            <div class="meta-row">
                                <strong>Verification Status:</strong>
                                <span class="status-pill ${escapeHtml(plate.status_class)}">${escapeHtml(plate.status_label)}</span>
                            </div>
                            <div class="meta-row">
                                <strong>Overall Score:</strong>
                                <span>${overallPercent}</span>
                            </div>
                            <div class="meta-row">
                                <strong>Detector Confidence:</strong>
                                <span>${detectorPercent}</span>
                            </div>
                            <div class="meta-row">
                                <strong>OCR Confidence:</strong>
                                <span>${ocrPercent}</span>
                            </div>
                            <div class="meta-row">
                                <strong>Singapore Checksum:</strong>
                                <span>${checksumText}</span>
                            </div>
                            <div class="meta-row">
                                <strong>Inference Source:</strong>
                                <span>${escapeHtml(plate.source)}</span>
                            </div>
                            <div class="meta-row">
                                <strong>Pipeline:</strong>
                                <span>${methodText}</span>
                            </div>
                            ${noteHtml}
                            ${displayNoteHtml}
                            <div class="meta-row">
                                <strong>Time Logged:</strong>
                                <span>${timestamp}</span>
                            </div>
                        </div>
                    </div>
                `;

                platesContainer.insertAdjacentHTML('beforeend', plateHtml);
            });
        } else {
            platesContainer.innerHTML = `
                <div class="empty-state">
                    <p>No readable number plate could be detected from this image.</p>
                    <p class="empty-state-sub">Try a clearer frame, brighter crop or a closer vehicle image.</p>
                </div>
            `;
        }

        resultsSection.classList.remove('hidden');
    }

    function formatPercent(value) {
        const numeric = Number(value || 0);
        return `${(numeric * 100).toFixed(1)}%`;
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    resetBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        fileInput.value = '';
        platesContainer.innerHTML = '';
        annotatedImage.src = '';
    });
});
