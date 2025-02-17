document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("feedbackForm");
    const outputStatus = document.querySelector(".output__message--status");
    const GOOGLE_FORM_BASE_URL = window.GOOGLE_FORM_URL; 
    const goBackButton = document.getElementById("go-back");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        
        try {
            // Get stored prefilled data
            const prefilledData = JSON.parse(localStorage.getItem('prefilledData'));
            
            if (!prefilledData) {
                throw new Error("No stored data found. Please complete the checker first.");
            }

            const rating = parseFloat(form.elements["rating"].value);
            const suggestedScore = parseFloat(form.elements["suggested_score"].value);

            // Validate inputs
            if (rating < 1 || rating > 5 || !Number.isInteger(rating)) {
                throw new Error("Rating must be a whole number between 1 and 5");
            }

            if (suggestedScore < 0 || suggestedScore > 1) {
                throw new Error("Suggested score must be between 0 and 1");
            }

            // Combine prefilled data with feedback
            const formData = new FormData();
            for (const [key, value] of Object.entries({
                ...prefilledData,
                'entry.749908925': rating,
                'entry.938705946': suggestedScore
            })) {
                formData.append(key, value);
            }
            // Send POST request to Google Forms endpoint
            const response = await fetch(GOOGLE_FORM_BASE_URL, {
                method: 'POST',
                mode: 'no-cors', // This is necessary for Google Forms as they don't allow CORS
                body: formData
            });

            if (response.ok || response.type === 'opaque') {
                outputStatus.textContent = "Feedback submitted successfully!";
                localStorage.removeItem('prefilledData');
                form.reset();
                goBackButton.style.display = 'block';
            } else {
                throw new Error("Failed to submit feedback.");
            }
        } catch (error) {
            outputStatus.textContent = error.message || "An error occurred while submitting feedback.";
            console.error("Error:", error);
        }
    });

    // Check if there's stored data when page loads
    const storedData = localStorage.getItem('prefilledData');
    if (!storedData) {
        outputStatus.textContent = "Please complete the checker first before providing feedback.";
        form.style.display = "none";
    }
});