// Main JavaScript for ICU Risk Prediction System

// Global variables
let currentTime = new Date();
let isUpdating = false;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCommonFeatures();
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000); // Update every second
});

function initializeCommonFeatures() {
    // Initialize tooltips if Bootstrap is loaded
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Add loading states to buttons
    addButtonLoadingStates();
    
    // Add smooth scrolling
    addSmoothScrolling();
}

function updateCurrentTime() {
    const timeElement = document.getElementById('current-time');
    if (timeElement) {
        currentTime = new Date();
        timeElement.textContent = currentTime.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
}

function addButtonLoadingStates() {
    // Add loading states to form submissions
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.disabled) {
                const originalText = submitBtn.innerHTML;
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="loading me-2"></span>Processing...';
                
                // Reset after 30 seconds if not reset by form handler
                setTimeout(() => {
                    if (submitBtn.disabled) {
                        submitBtn.disabled = false;
                        submitBtn.innerHTML = originalText;
                    }
                }, 30000);
            }
        });
    });
}

function addSmoothScrolling() {
    // Add smooth scrolling to anchor links
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            // Only scroll if href is not just '#'
            const href = this.getAttribute('href');
            if (href && href !== '#') {
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
}

// API call functions
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Trigger prediction function (called from templates)
async function triggerPrediction() {
    if (isUpdating) {
        showNotification('Update already in progress', 'warning');
        return;
    }

    isUpdating = true;
    const button = document.getElementById('refresh-btn') || event?.target;
    const originalText = button ? button.innerHTML : '';
    
    try {
        if (button) {
            button.disabled = true;
            button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Updating...';
        }
        
        const response = await apiCall('/api/trigger_prediction', {
            method: 'POST'
        });
        
        if (response.success) {
            showNotification('Predictions updated successfully!', 'success');
            // Reload the page to show new data
            setTimeout(() => {
                window.location.reload();
            }, 1000);
        } else {
            showNotification('Error updating predictions: ' + (response.message || response.error || 'Unknown error'), 'danger');
        }
    } catch (error) {
        showNotification('Network error: ' + error.message, 'danger');
    } finally {
        isUpdating = false;
        if (button) {
            button.disabled = false;
            button.innerHTML = originalText;
        }
    }
}

// Notification system
function showNotification(message, type = 'info', duration = 5000) {
    const notificationContainer = getOrCreateNotificationContainer();
    
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show notification-item`;
    notification.innerHTML = `
        <i class="fas fa-${getIconForType(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    notificationContainer.appendChild(notification);
    
    // Auto-dismiss
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
}

function getOrCreateNotificationContainer() {
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1050;
            max-width: 400px;
        `;
        document.body.appendChild(container);
    }
    return container;
}

function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// File upload validation
function validateFileUpload(input) {
    const file = input.files[0];
    if (!file) return false;
    
    const allowedTypes = [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'text/csv',
        'application/csv',
        'text/plain'
    ];
    
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!allowedTypes.includes(file.type)) {
        showNotification('Please select a valid Excel or CSV file (.xlsx, .xls, .csv)', 'danger');
        input.value = '';
        return false;
    }
    
    if (file.size > maxSize) {
        showNotification('File size must be less than 16MB', 'danger');
        input.value = '';
        return false;
    }
    
    return true;
}

// Add file validation to all file inputs
document.addEventListener('DOMContentLoaded', function() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            validateFileUpload(this);
        });
    });
});

// Format numbers for display
function formatNumber(num, decimals = 1) {
    if (num === null || num === undefined || isNaN(num)) return 'N/A';
    return parseFloat(num).toFixed(decimals);
}

function formatPercentage(num, decimals = 1) {
    if (num === null || num === undefined || isNaN(num)) return 'N/A';
    return (parseFloat(num) * 100).toFixed(decimals) + '%';
}

// Format dates
function formatDateTime(dateString) {
    if (!dateString) return 'N/A';
    
    try {
        const date = new Date(dateString);
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (error) {
        return 'Invalid Date';
    }
}

// Utility functions for risk calculations
function getRiskLevel(probability) {
    if (probability < 0.4) return 'Low';
    if (probability < 0.8) return 'Moderate';
    return 'High';
}

function getRiskColor(riskLevel) {
    const colors = {
        'Low': '#28a745',
        'Moderate': '#ffc107',
        'High': '#dc3545'
    };
    return colors[riskLevel] || '#6c757d';
}

function getRiskBadgeClass(riskLevel) {
    const classes = {
        'Low': 'bg-success',
        'Moderate': 'bg-warning',
        'High': 'bg-danger'
    };
    return classes[riskLevel] || 'bg-secondary';
}

// Load validation status
async function loadValidationStatus() {
    try {
        const response = await apiCall('/api/validation_status');
        
        if (response) {
            // Update validation metrics
            const successRateElement = document.getElementById('validation-success-rate');
            const totalPredictionsElement = document.getElementById('total-predictions');
            const validationErrorsElement = document.getElementById('validation-errors');
            const validationWarningsElement = document.getElementById('validation-warnings');
            
            if (successRateElement) {
                const rate = response.validation_success_rate || 0;
                successRateElement.textContent = `${rate.toFixed(1)}%`;
                
                // Color code the success rate
                if (rate >= 95) {
                    successRateElement.className = 'h4 mb-1 text-success';
                } else if (rate >= 80) {
                    successRateElement.className = 'h4 mb-1 text-warning';
                } else {
                    successRateElement.className = 'h4 mb-1 text-danger';
                }
            }
            
            if (totalPredictionsElement) {
                totalPredictionsElement.textContent = response.total_predictions || 0;
            }
            
            if (validationErrorsElement) {
                validationErrorsElement.textContent = response.failed_validations || 0;
            }
            
            if (validationWarningsElement) {
                validationWarningsElement.textContent = response.total_warnings || 0;
            }
        }
    } catch (error) {
        console.error('Error loading validation status:', error);
        showNotification('Error loading validation status', 'danger');
    }
}

// Show validation details modal
function showValidationDetails() {
    // This would show a modal with detailed validation information
    // For now, just show a notification
    showNotification('Validation details feature coming soon', 'info');
}

// Initialize validation status when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Load initial validation status
    loadValidationStatus();
    
    // Add event listeners for validation buttons
    const refreshValidationBtn = document.getElementById('refresh-validation');
    const viewDetailsBtn = document.getElementById('view-validation-details');
    
    if (refreshValidationBtn) {
        refreshValidationBtn.addEventListener('click', loadValidationStatus);
    }
    
    if (viewDetailsBtn) {
        viewDetailsBtn.addEventListener('click', showValidationDetails);
    }
});

// Export functions for use in other scripts
window.ICUPrediction = {
    triggerPrediction,
    showNotification,
    formatNumber,
    formatPercentage,
    formatDateTime,
    getRiskLevel,
    getRiskColor,
    getRiskBadgeClass,
    apiCall,
    loadValidationStatus,
    showValidationDetails
}; 