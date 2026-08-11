document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    
    const resultArea = document.getElementById('result-area');
    const imagePreview = document.getElementById('image-preview');
    const spinner = document.getElementById('loading-spinner');
    const resetBtn = document.getElementById('reset-btn');
    
    const predictionLabel = document.getElementById('prediction-label');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceFill = document.getElementById('confidence-fill');

    // --- Drag and Drop Events ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFiles(files);
        }
    }

    // --- Browse Button Events ---
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFiles(this.files);
        }
    });

    // --- File Handling and API Call ---
    function handleFiles(files) {
        const file = files[0];
        
        // Ensure it's an image
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            showResultArea();
            uploadImage(file);
        }
    }

    function showResultArea() {
        dropZone.classList.add('hidden');
        resultArea.classList.remove('hidden');
        
        // Reset UI state to loading
        spinner.classList.remove('hidden');
        predictionLabel.textContent = 'Analyzing...';
        predictionLabel.className = '';
        confidenceValue.textContent = '0%';
        confidenceFill.style.width = '0%';
        confidenceFill.className = 'progress-fill';
    }

    function resetUI() {
        resultArea.classList.add('hidden');
        dropZone.classList.remove('hidden');
        fileInput.value = '';
    }

    resetBtn.addEventListener('click', resetUI);

    async function uploadImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            displayResult(data);
        } catch (error) {
            console.error('Error:', error);
            spinner.classList.add('hidden');
            predictionLabel.textContent = 'Error occurred';
            predictionLabel.className = 'text-danger';
        }
    }

    function displayResult(data) {
        spinner.classList.add('hidden');
        
        const label = data.label;
        const confidence = data.confidence;
        const percentage = Math.round(confidence * 100);
        
        predictionLabel.textContent = label;
        confidenceValue.textContent = percentage + '%';
        
        // Animate progress bar
        setTimeout(() => {
            confidenceFill.style.width = percentage + '%';
        }, 100);

        // Apply colors based on prediction
        predictionLabel.className = '';
        confidenceFill.className = 'progress-fill';
        
        if (label === 'Rust') {
            predictionLabel.classList.add('text-danger');
            confidenceFill.classList.add('bg-danger');
        } else if (label === 'No rust') {
            predictionLabel.classList.add('text-success');
            confidenceFill.classList.add('bg-success');
        } else {
            // Can't classify
            predictionLabel.classList.add('text-warning');
            confidenceFill.classList.add('bg-warning');
        }
    }
});
