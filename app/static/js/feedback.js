document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("feedbackForm");
    const outputStatus = document.querySelector(".output__message--status");
    const GOOGLE_FORM_BASE_URL = window.GOOGLE_FORM_URL; 
    const goBackButton = document.getElementById("go-back");
    const backButton = document.getElementById("back-btn"); 
    
    // Function to send feedback
    const sendFeedback = async () => {
        try {
            // Get stored prefilled data
            const prefilledData = JSON.parse(localStorage.getItem('prefilledData'));
            
            if (!prefilledData) {
                throw new Error("No stored data found. Please complete the checker first.");
            }

            // Combine prefilled data without an issue field
            const formData = new FormData();
            for (const [key, value] of Object.entries(prefilledData)) {
                formData.append(key, value);
            }
            
            // Send POST request to Google Forms endpoint
            const response = await fetch(GOOGLE_FORM_BASE_URL, {
                method: 'POST',
                mode: 'no-cors', // This is necessary for Google Forms as they don't allow CORS
                body: formData
            });

            if (response.ok || response.type === 'opaque') {
                //outputStatus.textContent = "Feedback submitted successfully!";
                localStorage.removeItem('prefilledData');
                //goBackButton.style.display = 'block';
                return true;
            } else {
                throw new Error("Failed to submit feedback.");
            }
        } catch (error) {
            outputStatus.textContent = error.message || "An error occurred while submitting feedback.";
            console.error("Error:", error);
            return false;
        }
    };

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        
        const issue = form.elements["issue"].value;

        // Validate input
        if (!issue.trim()) {
            outputStatus.textContent = "Please describe the issue.";
            return;
        }

        // This is for form submission with an issue
        const success = await sendFeedbackWithIssue(issue);
        if (success) {
            form.reset();
        }
    });

    // Function to send feedback with an issue (for form submission)
    const sendFeedbackWithIssue = async (issue) => {
        try {
            const prefilledData = JSON.parse(localStorage.getItem('prefilledData'));
            
            if (!prefilledData) {
                throw new Error("No stored data found. Please complete the checker first.");
            }

            const formData = new FormData();
            for (const [key, value] of Object.entries({
                ...prefilledData,
                'entry.938705946': issue  // Issue field
            })) {
                formData.append(key, value);
            }
            
            const response = await fetch(GOOGLE_FORM_BASE_URL, {
                method: 'POST',
                mode: 'no-cors',
                body: formData
            });

            if (response.ok || response.type === 'opaque') {
                outputStatus.textContent = "Feedback submitted successfully!";
                localStorage.removeItem('prefilledData');
                goBackButton.style.display = 'block';
                return true;
            } else {
                throw new Error("Failed to submit feedback.");
            }
        } catch (error) {
            outputStatus.textContent = error.message || "An error occurred while submitting feedback.";
            console.error("Error:", error);
            return false;
        }
    };

    // Check if there's stored data when page loads
    const storedData = localStorage.getItem('prefilledData');
    if (!storedData) {
        outputStatus.textContent = "Please complete the checker first before providing feedback.";
        form.style.display = "none";
    } else {
        const data = JSON.parse(storedData);
        // Display names and rating
        document.querySelector('input[name="name1"]').value = data["entry.2072591192"]; // candidate name
        document.querySelector('input[name="name2"]').value = data["entry.872788899"]; // query name
        document.querySelector('input[name="rating"]').value = (parseFloat(data["entry.749418275"]) * 100).toFixed(2); // rating
    }

    // Back button event listener - sends feedback without an issue
    backButton.addEventListener('click', async (event) => {
        event.preventDefault(); // Prevent default action of the link
        const success = await sendFeedback();
        if (success) {
            // If feedback is sent successfully, then proceed with navigation
            window.location.href = backButton.href;
        }
    });
});