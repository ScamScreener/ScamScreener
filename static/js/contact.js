// Function to handle the form submission
function submitContactForm(event) {
    event.preventDefault();
    
    const submitBtn = document.querySelector('.submit-btn');
    const originalText = submitBtn.textContent;
    
    submitBtn.textContent = 'Sending...';
    submitBtn.disabled = true;
    
    setTimeout(() => {
        alert('Thank you for your message! We\'ll get back to you within 24 hours.');
        
        // Reset form
        event.target.reset();
        
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    }, 2000);
}

// Add smooth animations to contact cards
document.querySelectorAll('.contact-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-10px)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
    });
});

// Add focus effects to form inputs
document.querySelectorAll('.form-input, .form-textarea, .form-select').forEach(input => {
    input.addEventListener('focus', function() {
        this.style.transform = 'scale(1.02)';
    });
    
    input.addEventListener('blur', function() {
        this.style.transform = 'scale(1)';
    });
});
