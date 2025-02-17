document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");
  const outputMessageFirst = document.querySelector(".output__message--first");
  const outputMessageSecond = document.querySelector(".output__message--second");
  const outputMessageThird = document.querySelector(".output__message--third");
  const outputMessageFourth= document.querySelector(".output__message--fourth");
  const outputStatus = document.querySelector(".output__message--status");
  const feedbackButton = document.querySelector(".feedback-button");
  const cancelButton = document.querySelector(".cancel-button");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const name1 = form.elements["name1"].value.trim();
    const name2 = form.elements["name2"].value.trim();

    if (name1 && name2) {
      try {
        // Send data to the backend using fetch
        const response = await fetch("/process", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ name1, name2 }),
        });

        if (!response.ok) {
          throw new Error("Network response was not ok");
        }

        const data = await response.json();

        // Display the result
        outputMessageFirst.textContent = `Name 1: ${data.match.candidate_name}`;
        outputMessageSecond.textContent = `Name 2: ${data.match.query_name}`;
        outputMessageThird.textContent = `Score: ${(data.match.final_score * 100).toFixed(2)}%`;
        outputMessageFourth.textContent = `Decision: ${data.match.confidence}`;
        feedbackButton.style.display = "block";
        cancelButton.style.display = 'block';

        localStorage.setItem('prefilledData', JSON.stringify(data.prefilled_data));

        // Update feedback button link
        feedbackButton.onclick = () => {
          window.location.href = '/feedback';
        };
      } catch (error) {
        outputStatus.textContent = "An error occurred while processing.";
        console.error("Error:", error);
      }
    } else {
      outputStatus.textContent = "Please fill out both names.";
      feedbackButton.style.display = "none";
      cancelButton.style.display = 'none';
    }
  });

  // Function to reset the state after sending feedback to Google Form
  const resetState = async () => {
    const prefilledData = JSON.parse(localStorage.getItem('prefilledData'));
    if (prefilledData) {
      try {
       
        const GOOGLE_FORM_BASE_URL = window.GOOGLE_FORM_URL;
        
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
          outputStatus.textContent = "Feedback submitted!";
          setTimeout(() => {
            outputStatus.style.display = "none";
          }, 3000)
        
        } else {
          throw new Error("Failed to submit feedback.");
        }
      } catch (error) {
        outputStatus.textContent = "Feedback submission failed.";
        console.error("Feedback Error:", error);
      } finally {
        // Reset state regardless of feedback submission success
        form.reset();
        localStorage.removeItem('prefilledData');
        outputMessageFirst.textContent = '';
        outputMessageSecond.textContent = '';
        outputMessageThird.textContent = '';
        outputMessageFourth.textContent = '';
        feedbackButton.style.display = 'none';
        cancelButton.style.display = 'none';
      }
    } else {
      form.reset();
      outputMessageFirst.textContent = '';
      outputMessageSecond.textContent = '';
      outputMessageThird.textContent = '';
      outputMessageFourth.textContent = '';
      outputStatus.textContent = '';
      feedbackButton.style.display = 'none';
      cancelButton.style.display = 'none';
    }
  };

  // Handle the cancel button click
  cancelButton.addEventListener("click", async (event) => {
      event.preventDefault();
      cancelButton.textContent = 'Resetting...';
      await resetState();  
      cancelButton.textContent = 'Reset'
  });

  feedbackButton.style.display = 'none';
  cancelButton.style.display = 'none';
});