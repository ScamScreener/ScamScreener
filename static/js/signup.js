document.addEventListener('DOMContentLoaded', function() {
  const signupForm = document.getElementById('signupForm');
  
  if (signupForm) {
    signupForm.addEventListener('submit', async function(e) {
      e.preventDefault();
      
      const username = document.getElementById('signupUsername').value.trim();
      const email = document.getElementById('signupEmail').value.trim();
      const firstName = document.getElementById('signupFirstName').value.trim();
      const lastName = document.getElementById('signupLastName').value.trim();
      const password = document.getElementById('signupPassword').value.trim();
      const errorMessage = document.getElementById('errorMessage');

      // Reset error message
      errorMessage.style.display = 'none';
      
      if (!username || !email || !password) {
        errorMessage.textContent = 'Please fill in all required fields';
        errorMessage.style.display = 'block';
        return;
      }

      // Email validation
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(email)) {
        errorMessage.textContent = 'Please enter a valid email address';
        errorMessage.style.display = 'block';
        return;
      }

      try {
        const response = await fetch('http://localhost:5000/auth/signup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            username,
            email,
            password,
            first_name: firstName || null,
            last_name: lastName || null
          })
        });

        const data = await response.json();

        if (response.ok) {
          alert('✅ Account created successfully! You can now log in.');
          window.location.href = 'login.html';
        } else {
          errorMessage.textContent = data.error || 'Signup failed';
          errorMessage.style.display = 'block';
        }
      } catch (error) {
        errorMessage.textContent = 'Network error. Please try again later.';
        errorMessage.style.display = 'block';
        console.error('Signup error:', error);
      }
    });
  }
});
