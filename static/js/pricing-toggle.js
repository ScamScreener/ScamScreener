// Pricing toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    let isYearly = false;
    const billingToggle = document.getElementById('billing-toggle');
    const proPrice = document.getElementById('pro-price');
    const enterprisePrice = document.getElementById('enterprise-price');
    
    // Add event listener to billing toggle
    if (billingToggle) {
        billingToggle.addEventListener('click', function() {
            isYearly = !isYearly;
            this.classList.toggle('active');
            
            if (isYearly) {
                proPrice.textContent = '15';
                enterprisePrice.textContent = '79';
            } else {
                proPrice.textContent = '19';
                enterprisePrice.textContent = '99';
            }
        });
    }
    
    // Add hover effects to pricing cards
    const pricingCards = document.querySelectorAll('.pricing-card');
    pricingCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});
