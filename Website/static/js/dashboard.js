// Dashboard-specific JavaScript for ICU Risk Prediction System

let riskChart = null;
// REMOVE: let predictionsData = [];

// Initialize dashboard features
function initializeDashboard(patientType, predictions) {
    // REMOVE: predictionsData = predictions || [];
    updateRiskDistribution(predictions);
    initializePatientSearch();
    initializeTableSorting();
    updateStatsSummary();
}

// Update risk distribution chart
function updateRiskDistribution(predictions) {
    if (!predictions || predictions.length === 0) {
        document.getElementById('low-risk-count').textContent = '0';
        document.getElementById('moderate-risk-count').textContent = '0';
        document.getElementById('high-risk-count').textContent = '0';
        document.getElementById('total-patients').textContent = '0';
        return;
    }

    // Calculate risk distribution based on risk_level labels
    const riskCounts = {
        low: predictions.filter(p => (p.risk_level || '').startsWith('Low Risk')).length,
        moderate: predictions.filter(p => (p.risk_level || '').startsWith('Medium Risk')).length,
        high: predictions.filter(p => (p.risk_level || '').startsWith('High Risk')).length
    };

    // Update risk count displays
    document.getElementById('low-risk-count').textContent = riskCounts.low;
    document.getElementById('moderate-risk-count').textContent = riskCounts.moderate;
    document.getElementById('high-risk-count').textContent = riskCounts.high;
    document.getElementById('total-patients').textContent = predictions.length;

    // Update or create chart
    const ctx = document.getElementById('riskDistributionChart');
    if (ctx) {
        if (riskChart) {
            riskChart.destroy();
        }

        riskChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk (Safe)', 'Medium Risk (Monitoring)', 'High Risk (Urgent)'],
                datasets: [{
                    data: [riskCounts.low, riskCounts.moderate, riskCounts.high],
                    backgroundColor: ['#28a745', '#ffc107', '#dc3545'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed * 100) / total).toFixed(1);
                                return `${context.label}: ${context.parsed} (${percentage}%)`;
                            }
                        }
                    }
                },
                cutout: '50%',
                animation: {
                    animateRotate: true,
                    duration: 1000
                }
            }
        });
    }
}

// Initialize patient search functionality
function initializePatientSearch() {
    const searchInput = document.getElementById('patient-search');
    const tableBody = document.getElementById('predictions-tbody');
    
    if (searchInput && tableBody) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase().trim();
            const rows = tableBody.querySelectorAll('tr');
            
            rows.forEach(row => {
                const searchableText = Array.from(row.cells).map(cell => 
                    cell.textContent.toLowerCase()
                ).join(' ');
                
                if (searchableText.includes(searchTerm)) {
                    row.style.display = '';
                    row.classList.add('prediction-row-new');
                    setTimeout(() => row.classList.remove('prediction-row-new'), 500);
                } else {
                    row.style.display = 'none';
                }
            });
            
            // Update visible count
            const visibleRows = Array.from(rows).filter(row => row.style.display !== 'none');
            updateSearchResults(visibleRows.length, rows.length);
        });
        
        // Add clear search button
        addClearSearchButton(searchInput);
    }
}

function addClearSearchButton(searchInput) {
    const clearButton = document.createElement('button');
    clearButton.type = 'button';
    clearButton.className = 'btn btn-outline-secondary btn-sm';
    clearButton.innerHTML = '<i class="fas fa-times"></i>';
    clearButton.style.cssText = 'position: absolute; right: 45px; top: 50%; transform: translateY(-50%); z-index: 10;';
    clearButton.style.display = 'none';
    
    clearButton.addEventListener('click', function() {
        searchInput.value = '';
        searchInput.dispatchEvent(new Event('input'));
        this.style.display = 'none';
    });
    
    searchInput.addEventListener('input', function() {
        clearButton.style.display = this.value ? 'block' : 'none';
    });
    
    const inputGroup = searchInput.closest('.input-group');
    if (inputGroup) {
        inputGroup.style.position = 'relative';
        inputGroup.appendChild(clearButton);
    }
}

function updateSearchResults(visible, total) {
    let resultDiv = document.getElementById('search-results');
    if (!resultDiv) {
        resultDiv = document.createElement('div');
        resultDiv.id = 'search-results';
        resultDiv.className = 'text-muted small mt-2';
        const searchInput = document.getElementById('patient-search');
        if (searchInput) {
            searchInput.parentNode.appendChild(resultDiv);
        }
    }
    
    if (visible < total) {
        resultDiv.textContent = `Showing ${visible} of ${total} patients`;
        resultDiv.style.display = 'block';
    } else {
        resultDiv.style.display = 'none';
    }
}

// Initialize table sorting
function initializeTableSorting() {
    const table = document.getElementById('predictions-table');
    if (!table) return;
    
    const headers = table.querySelectorAll('th');
    headers.forEach((header, index) => {
        // Skip non-sortable columns (all columns are sortable now)
        // No need to skip any columns
        
        header.style.cursor = 'pointer';
        header.addEventListener('click', () => sortTable(index));
        
        // Add sort icons
        header.innerHTML += ' <i class="fas fa-sort text-muted"></i>';
    });
}

function sortTable(columnIndex) {
    const table = document.getElementById('predictions-table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    // Determine sort direction
    const header = table.querySelectorAll('th')[columnIndex];
    const currentDirection = header.dataset.sortDirection || 'asc';
    const newDirection = currentDirection === 'asc' ? 'desc' : 'asc';
    
    // Clear all sort indicators
    table.querySelectorAll('th i').forEach(icon => {
        icon.className = 'fas fa-sort text-muted';
    });
    
    // Set new sort indicator
    const icon = header.querySelector('i');
    icon.className = `fas fa-sort-${newDirection === 'asc' ? 'up' : 'down'} text-primary`;
    header.dataset.sortDirection = newDirection;
    
    // Sort rows
    rows.sort((a, b) => {
        const aValue = getCellValue(a.cells[columnIndex]);
        const bValue = getCellValue(b.cells[columnIndex]);
        
        const result = aValue.localeCompare(bValue, undefined, { numeric: true });
        return newDirection === 'asc' ? result : -result;
    });
    
    // Reorder rows in DOM
    rows.forEach(row => tbody.appendChild(row));
}

function getCellValue(cell) {
    // Handle special cases like percentages and risk levels
    const text = cell.textContent.trim();
    
    // Extract numeric values from percentages
    if (text.includes('%')) {
        const match = text.match(/(\d+\.?\d*)%/);
        return match ? match[1] : text;
    }
    
    // Handle risk levels
    if (text.startsWith('Low Risk')) {
        return '1';
    } else if (text.startsWith('Medium Risk')) {
        return '2';
    } else if (text.startsWith('High Risk')) {
        return '3';
    }
    
    return text;
}

// Update statistics summary
function updateStatsSummary() {
    if (!predictionsData || predictionsData.length === 0) return;
    
    const stats = calculatePredictionStats(predictionsData);
    updateStatsDisplay(stats);
}

function calculatePredictionStats(predictions) {
    const totalPatients = predictions.length;
    const highRiskCount = predictions.filter(p => p.icu_risk_probability >= 0.8).length;
    const averageRisk = predictions.reduce((sum, p) => sum + p.icu_risk_probability, 0) / totalPatients;
    
    // Model distribution
    const modelDistribution = {};
    predictions.forEach(p => {
        modelDistribution[p.model_used] = (modelDistribution[p.model_used] || 0) + 1;
    });
    
    // Admission order distribution
    const admissionOrderDistribution = {};
    predictions.forEach(p => {
        admissionOrderDistribution[p.admission_order] = (admissionOrderDistribution[p.admission_order] || 0) + 1;
    });
    
    return {
        totalPatients,
        highRiskCount,
        highRiskPercentage: (highRiskCount / totalPatients) * 100,
        averageRisk: averageRisk * 100,
        modelDistribution,
        admissionOrderDistribution
    };
}

function updateStatsDisplay(stats) {
    // Create or update stats display
    let statsContainer = document.getElementById('stats-summary');
    if (!statsContainer) {
        statsContainer = document.createElement('div');
        statsContainer.id = 'stats-summary';
        statsContainer.className = 'alert alert-info mt-3';
        
        const riskSection = document.querySelector('.risk-summary-card').closest('.row');
        if (riskSection) {
            riskSection.parentNode.insertBefore(statsContainer, riskSection.nextSibling);
        }
    }
    
    statsContainer.innerHTML = `
        <h6><i class="fas fa-chart-line me-2"></i>Quick Statistics</h6>
        <div class="row">
            <div class="col-md-3">
                <strong>High Risk Patients:</strong><br>
                <span class="text-danger">${stats.highRiskCount} (${stats.highRiskPercentage.toFixed(1)}%)</span>
            </div>
            <div class="col-md-3">
                <strong>Average Risk:</strong><br>
                <span class="text-warning">${stats.averageRisk.toFixed(1)}%</span>
            </div>
            <div class="col-md-3">
                <strong>Hematology Model:</strong><br>
                <span class="text-info">${stats.modelDistribution.hematology || 0} patients</span>
            </div>
            <div class="col-md-3">
                <strong>Solid Model:</strong><br>
                <span class="text-secondary">${stats.modelDistribution.solid || 0} patients</span>
            </div>
        </div>
    `;
}

function getMostCommonAdmissionOrder(admissionOrderDistribution) {
    let maxCount = 0;
    let mostCommon = 'N/A';
    
    for (const [order, count] of Object.entries(admissionOrderDistribution)) {
        if (count > maxCount) {
            maxCount = count;
            mostCommon = order;
        }
    }
    
    return mostCommon;
}

// Refresh predictions from server
async function refreshPredictions(patientType) {
    try {
        const response = await ICUPrediction.apiCall(`/api/predictions/${patientType}`);
        if (response.predictions) {
            predictionsData = response.predictions;
            updateRiskDistribution(predictionsData);
            updatePredictionTable(predictionsData);
            updateStatsSummary();
            
            // Update last update time
            const lastUpdateElement = document.getElementById('last-update-time');
            if (lastUpdateElement && response.last_update) {
                lastUpdateElement.textContent = ICUPrediction.formatDateTime(response.last_update);
            }
        }
    } catch (error) {
        console.error('Failed to refresh predictions:', error);
        ICUPrediction.showNotification('Failed to refresh predictions', 'warning');
    }
}

function updatePredictionTable(predictions) {
    const tbody = document.getElementById('predictions-tbody');
    if (!tbody) return;
    
    // Store current search term
    const searchInput = document.getElementById('patient-search');
    const currentSearch = searchInput ? searchInput.value : '';
    
    // Clear existing rows
    tbody.innerHTML = '';
    
    // Add new rows
    predictions.forEach(patient => {
        const row = createPredictionRow(patient);
        tbody.appendChild(row);
    });
    
    // Reapply search if there was one
    if (currentSearch && searchInput) {
        searchInput.value = currentSearch;
        searchInput.dispatchEvent(new Event('input'));
    }
    
    // Update total patient count
    const totalElement = document.getElementById('total-patients');
    if (totalElement) {
        totalElement.textContent = predictions.length;
    }
}

function createPredictionRow(patient) {
    const row = document.createElement('tr');
    row.dataset.risk = patient.risk_level.toLowerCase();
    row.className = 'prediction-row-new';
    
    const riskPercentage = (patient.icu_risk_probability * 100).toFixed(1);
    const riskLevel = patient.risk_level || '';
    const progressBarClass = riskLevel.startsWith('Low Risk') ? 'bg-success' :
                             riskLevel.startsWith('Medium Risk') ? 'bg-warning' : 'bg-danger';
    const badgeClass = progressBarClass;
    const cohortLabel = patient.model_used === 'hematology' ? 'Hematology' : 'Non-Hematology';
    const cohortClass = patient.model_used === 'hematology' ? 'bg-danger' : 'bg-light text-dark border';
    
    row.innerHTML = `
        <td><strong>${patient.patient_name}</strong></td>
        <td>${patient.mrn}</td>
        <td>${patient.location}</td>
        <td>${patient.room}</td>
        <td><span class="badge ${cohortClass}">${cohortLabel}</span></td>
        <td>${patient.admission_date || ''}</td>
        <td>
            <div class="d-flex align-items-center">
                <div class="progress me-2" style="width: 100px; height: 20px;">
                    <div class="progress-bar ${progressBarClass}" style="width: ${riskPercentage}%"></div>
                </div>
                <span class="fw-bold">${riskPercentage}%</span>
            </div>
        </td>
        <td>
            <span class="badge ${badgeClass}">${patient.risk_level}</span>
        </td>
        <td>
            <div class="btn-group" role="group">
                <button class="btn btn-sm btn-outline-success rounded-pill px-3 feedback-btn" title="Looks good" data-feedback="up" aria-label="Thumbs up">
                    <i class="fas fa-thumbs-up"></i>
                </button>
                <button class="btn btn-sm btn-outline-danger rounded-pill px-3 feedback-btn" title="Needs attention" data-feedback="down" aria-label="Thumbs down">
                    <i class="fas fa-thumbs-down"></i>
                </button>
            </div>
        </td>
    `;
    
    // Remove animation class after animation completes
    setTimeout(() => row.classList.remove('prediction-row-new'), 500);
    
    return row;
}

// Export functions for global use
window.updateRiskDistribution = updateRiskDistribution;
window.initializePatientSearch = initializePatientSearch;
window.refreshPredictions = refreshPredictions; 