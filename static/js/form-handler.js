document.getElementById('trialForm').addEventListener('submit', async function(e) {
  e.preventDefault();

  const name = document.getElementById('trialName').value.trim();
  const email = document.getElementById('trialEmail').value.trim();
  const password = document.getElementById('trialPassword').value.trim();
  const successMessage = document.getElementById('successMessage');
  const errorMessage = document.getElementById('errorMessage');

  try {
    const response = await fetch('http://localhost:5000/auth/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: email.split("@")[0] + "_" + Date.now(), // generate username from email
        email: email,
        password: password,
        first_name: name
      })
    });

    const data = await response.json();

    if (response.ok) {
      successMessage.style.display = 'block';
      errorMessage.style.display = 'none';
      e.target.reset();
    } else {
      errorMessage.textContent = data.error || "Failed to create trial account";
      errorMessage.style.display = 'block';
    }
  } catch (err) {
    errorMessage.textContent = `Error: ${err.message}`;
    errorMessage.style.display = 'block';
  }
});
