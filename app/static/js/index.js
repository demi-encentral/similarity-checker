document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");
    const outputMessageFirst = document.querySelector(".output__message--first");
    const outputMessageSecond = document.querySelector(".output__message--second");
    const outputMessageThird = document.querySelector(".output__message--third");
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
          outputMessageFirst.textContent = `Candidate Name: ${data.match.candidate_name}`;
          outputMessageSecond.textContent = `Query Name: ${data.match.query_name}`;
          outputMessageThird.textContent = `Score: ${data.match.final_score.toFixed(2)}`;
          outputStatus.textContent = `Decision: ${data.match.confidence}`;
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

    
    // Function to reset the state
    const resetState = () => {
    // Clear all inputs
    form.reset();
    
    // Clear local storage
    localStorage.removeItem('prefilledData');
    
    // Reset all display elements
    outputMessageFirst.textContent = '';
    outputMessageSecond.textContent = '';
    outputMessageThird.textContent = '';
    outputStatus.textContent = '';
    
    // Hide buttons
    feedbackButton.style.display = 'none';
    cancelButton.style.display = 'none';
    };
    // Handle the cancel button click
    cancelButton.addEventListener("click", (event) => {
        event.preventDefault();
        resetState();
    });
    // Show/hide buttons when needed
    feedbackButton.style.display = 'none';
    cancelButton.style.display = 'none';
    
  });
  