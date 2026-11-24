// =====================================================
// FOODWISE AI - FRONTEND JAVASCRIPT
// =====================================================

const API_BASE = "http://127.0.0.1:5000";

let categoryChart = null;
let predictionChart = null;
let inventoryRecords = [];
let donationRecords = [];

// =====================================================
// GENERAL HELPERS
// =====================================================

function byId(id) {
    return document.getElementById(id);
}

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    })[character]);
}

async function apiRequest(url, options = {}) {
    const response = await fetch(`${API_BASE}${url}`, {
        ...options,
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {})
        }
    });

    const result = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(
            result.message ||
            result.error ||
            `Request failed with status ${response.status}`
        );
    }

    return result;
}

function showMessage(elementId, message, type = "success") {
    const element = byId(elementId);

    element.textContent = message;
    element.className = `message-box message-${type}`;
    element.classList.remove("hidden");

    if (type === "success") {
        setTimeout(() => element.classList.add("hidden"), 5000);
    }
}

function formatNumber(value) {
    return Number(value || 0).toLocaleString("en-IN", {
        maximumFractionDigits: 2
    });
}

function formatDate(value) {
    if (!value) return "-";
    return String(value).split("T")[0];
}

function localDateString() {
    const today = new Date();

    return [
        today.getFullYear(),
        String(today.getMonth() + 1).padStart(2, "0"),
        String(today.getDate()).padStart(2, "0")
    ].join("-");
}

function setDefaultDates() {
    const today = new Date();

    byId("todayDate").textContent = today.toLocaleDateString("en-IN", {
        day: "numeric",
        month: "short",
        year: "numeric"
    });

    byId("predictionDate").value = localDateString();
    byId("purchaseDate").value = localDateString();
    byId("donationDate").value = localDateString();

    byId("predictionDate").max = localDateString();
}

// =====================================================
// SIDEBAR NAVIGATION
// =====================================================

document.querySelectorAll(".nav-link").forEach(button => {
    button.addEventListener("click", () => {
        const sectionId = button.dataset.section;

        document.querySelectorAll(".nav-link").forEach(link =>
            link.classList.remove("active")
        );

        document.querySelectorAll(".page-section").forEach(section =>
            section.classList.remove("active")
        );

        button.classList.add("active");
        byId(sectionId).classList.add("active");


        // Reset scroll position when changing pages
        const mainContent = document.querySelector(".main-content");

        if (mainContent) {
            mainContent.scrollTo({
                top: 0,
                behavior: "auto"
            });
        }

        window.scrollTo(0, 0);


        const titles = {
            dashboard: "Dashboard",
            prediction: "Waste Prediction",
            demand: "Demand Forecasting",
            inventory: "Inventory Management",
            history: "Prediction History",
            donations: "Food Donations",
            about: "About FoodWise AI"
        };

        byId("pageTitle").textContent = titles[sectionId] || "FoodWise AI";

        if (sectionId === "inventory") loadInventory();
        if (sectionId === "history") loadPredictionHistory();
        if (sectionId === "donations") loadDonations();
    });
});

// =====================================================
// BACKEND CONNECTION
// =====================================================

async function checkBackend() {
    const status = byId("connectionStatus");
    const dot = document.querySelector(".status-dot");

    try {
        await apiRequest("/api/health");

        status.textContent = "Connected";
        dot.style.background = "#42d982";
    } catch (error) {
        status.textContent = "Disconnected";
        dot.style.background = "#e74c3c";
        console.error("Backend connection error:", error);
    }
}

// =====================================================
// DASHBOARD ANALYTICS
// =====================================================

async function loadDashboardStats() {
    try {
        const data = await apiRequest("/api/dashboard-stats");

        const dataset = data.dataset_statistics;
        const predictions = data.prediction_statistics;

        byId("totalWaste").textContent =
            formatNumber(dataset.total_food_waste);

        byId("categories").textContent =
            formatNumber(dataset.food_categories);

        byId("totalPredictions").textContent =
            formatNumber(predictions.total_predictions);

        byId("highPredictions").textContent =
            formatNumber(predictions.high_predictions);

        drawCategoryChart(data.waste_by_category || []);
        drawPredictionChart(predictions);

    } catch (error) {
        console.error("Dashboard statistics error:", error);

        byId("totalWaste").textContent = "Error";
        byId("categories").textContent = "--";
        byId("totalPredictions").textContent = "--";
        byId("highPredictions").textContent = "--";
    }
}

function drawCategoryChart(categories) {
    const canvas = byId("categoryChart");

    if (categoryChart) categoryChart.destroy();

    categoryChart = new Chart(canvas, {
        type: "bar",
        data: {
            labels: categories.map(item => item.category),
            datasets: [{
                label: "Waste Quantity",
                data: categories.map(item => item.waste_quantity),
                backgroundColor: [
                    "#318b61",
                    "#e6a34b",
                    "#5489c8",
                    "#65a87a",
                    "#c97868",
                    "#9b83c7"
                ],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { precision: 0 }
                }
            }
        }
    });
}

function drawPredictionChart(stats) {
    const canvas = byId("predictionChart");

    if (predictionChart) predictionChart.destroy();

    predictionChart = new Chart(canvas, {
        type: "doughnut",
        data: {
            labels: ["High", "Medium", "Low"],
            datasets: [{
                data: [
                    stats.high_predictions || 0,
                    stats.medium_predictions || 0,
                    stats.low_predictions || 0
                ],
                backgroundColor: ["#dc5555", "#e7aa43", "#35a56d"],
                borderWidth: 2,
                borderColor: "#ffffff"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom" }
            }
        }
    });
}

// =====================================================
// WASTE PREDICTION
// =====================================================

byId("predictionForm").addEventListener("submit", async event => {
    event.preventDefault();

    const submitButton = event.submitter;
    const originalText = submitButton.textContent;

    submitButton.disabled = true;
    submitButton.textContent = "Generating...";

    const payload = {
        business_type: byId("businessType").value,
        food_category: byId("foodCategory").value,
        purchase_quantity: byId("purchaseQuantity").value,
        shelf_life_days: byId("shelfLife").value,
        storage_temperature: byId("storageTemperature").value,
        promotion: byId("promotion").value,
        weather: byId("weather").value,
        date: byId("predictionDate").value
    };

    try {
        const data = await apiRequest("/api/predict-waste", {
            method: "POST",
            body: JSON.stringify(payload)
        });

        const result = byId("predictionResult");
        const level = data.prediction;

        result.innerHTML = `
            <h3>Prediction Generated Successfully</h3>
            <p><strong>Predicted Waste Level:</strong>
                ${wasteBadge(level)}
            </p>
            <p><strong>High Probability:</strong> ${escapeHTML(data.probabilities.High)}%</p>
            <p><strong>Medium Probability:</strong> ${escapeHTML(data.probabilities.Medium)}%</p>
            <p><strong>Low Probability:</strong> ${escapeHTML(data.probabilities.Low)}%</p>
            
            <p class="notice">${escapeHTML(data.notice)}</p>
        `;

        result.classList.remove("hidden");

        await Promise.all([
            loadDashboardStats(),
            loadPredictionHistory()
        ]);

    } catch (error) {
        const result = byId("predictionResult");
        result.textContent = `Prediction failed: ${error.message}`;
        result.classList.remove("hidden");
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = originalText;
    }
});

// =====================================================
// PREDICTION HISTORY
// =====================================================

async function loadPredictionHistory() {
    const historyBody = byId("historyTableBody");
    const recentBody = byId("dashboardRecentBody");

    try {
        const data = await apiRequest("/api/prediction-history?limit=200");
        const predictions = data.predictions || [];

        if (predictions.length === 0) {
            historyBody.innerHTML =
                `<tr><td colspan="9" class="empty-state">No predictions found.</td></tr>`;

            recentBody.innerHTML =
                `<tr><td colspan="5" class="empty-state">No predictions found.</td></tr>`;

            return;
        }

        historyBody.innerHTML = predictions.map(item => `
            <tr>
                <td>${escapeHTML(item.prediction_id)}</td>
                <td>${escapeHTML(item.business_type)}</td>
                <td>${escapeHTML(item.food_category)}</td>
                <td>${formatNumber(item.purchase_quantity)}</td>
                <td>${wasteBadge(item.predicted_waste_level)}</td>
                <td>${formatNumber(item.high_probability)}%</td>
                <td>${formatNumber(item.medium_probability)}%</td>
                <td>${formatNumber(item.low_probability)}%</td>
                <td>${escapeHTML(formatDate(item.prediction_date))}</td>
            </tr>
        `).join("");

        recentBody.innerHTML = predictions.slice(0, 5).map(item => `
            <tr>
                <td>${escapeHTML(item.prediction_id)}</td>
                <td>${escapeHTML(item.business_type)}</td>
                <td>${escapeHTML(item.food_category)}</td>
                <td>${wasteBadge(item.predicted_waste_level)}</td>
                <td>${escapeHTML(formatDate(item.prediction_date))}</td>
            </tr>
        `).join("");

    } catch (error) {
        console.error("Prediction history error:", error);

        historyBody.innerHTML =
            `<tr><td colspan="9" class="empty-state">Could not load prediction history.</td></tr>`;

        recentBody.innerHTML =
            `<tr><td colspan="5" class="empty-state">Could not load predictions.</td></tr>`;
    }
}

function wasteBadge(level) {
    let className = "badge-low-waste";

    if (level === "High") className = "badge-high";
    if (level === "Medium") className = "badge-medium";

    return `<span class="badge ${className}">${escapeHTML(level)}</span>`;
}

// =====================================================
// INVENTORY MANAGEMENT
// =====================================================

async function loadInventory() {
    const tableBody = byId("inventoryTableBody");

    tableBody.innerHTML =
        `<tr><td colspan="10" class="empty-state">Loading inventory...</td></tr>`;

    try {
        const data = await apiRequest("/api/inventory");
        inventoryRecords = data.inventory || [];
        renderInventory(inventoryRecords);

    } catch (error) {
        console.error("Inventory loading error:", error);

        tableBody.innerHTML =
            `<tr><td colspan="10" class="empty-state">Could not load inventory: ${escapeHTML(error.message)}</td></tr>`;
    }
}

function getStockStatus(item) {
    if (Number(item.quantity) <= Number(item.reorder_level)) {
        return { label: "Low Stock", className: "badge-low" };
    }

    return { label: "Normal", className: "badge-normal" };
}

function getExpiryStatus(item) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const expiry = new Date(`${formatDate(item.expiry_date)}T00:00:00`);
    const daysRemaining = Math.ceil(
        (expiry - today) / (1000 * 60 * 60 * 24)
    );

    if (daysRemaining < 0) {
        return {
            label: "Expired",
            className: "badge-expired",
            days: daysRemaining
        };
    }

    if (daysRemaining <= 3) {
        return {
            label: daysRemaining === 0 ? "Expires Today" : `In ${daysRemaining} day(s)`,
            className: "badge-expiring",
            days: daysRemaining
        };
    }

    return {
        label: "Safe",
        className: "badge-safe",
        days: daysRemaining
    };
}

function renderInventory(items) {
    const tableBody = byId("inventoryTableBody");

    byId("inventoryCount").textContent = items.length;

    const lowStockItems = items.filter(item =>
        Number(item.quantity) <= Number(item.reorder_level)
    );

    const expiredItems = items.filter(item =>
        getExpiryStatus(item).days < 0
    );

    const expiringItems = items.filter(item => {
        const days = getExpiryStatus(item).days;
        return days >= 0 && days <= 3;
    });

    byId("lowStockCount").textContent = lowStockItems.length;
    byId("expiredCount").textContent = expiredItems.length;
    byId("expiringCount").textContent = expiringItems.length;

    renderInventoryAlerts(lowStockItems, expiredItems, expiringItems);

    if (items.length === 0) {
        tableBody.innerHTML =
            `<tr><td colspan="10" class="empty-state">No inventory items yet. Add your first item using the form above.</td></tr>`;
        return;
    }

    tableBody.innerHTML = items.map(item => {
        const stock = getStockStatus(item);
        const expiry = getExpiryStatus(item);

        return `
            <tr>
                <td>${escapeHTML(item.inventory_id)}</td>
                <td><strong>${escapeHTML(item.item_name)}</strong></td>
                <td>${escapeHTML(item.food_category)}</td>
                <td>${escapeHTML(item.business_type)}</td>
                <td>${formatNumber(item.quantity)} ${escapeHTML(item.unit)}</td>
                <td>${escapeHTML(formatDate(item.purchase_date))}</td>
                <td>${escapeHTML(formatDate(item.expiry_date))}</td>
                <td><span class="badge ${stock.className}">${stock.label}</span></td>
                <td><span class="badge ${expiry.className}">${escapeHTML(expiry.label)}</span></td>
                <td>
                    <div class="action-buttons">
                        <button class="btn btn-outline btn-small"
                            data-action="edit"
                            data-id="${Number(item.inventory_id)}">Edit</button>
                        <button class="btn btn-danger btn-small"
                            data-action="delete"
                            data-id="${Number(item.inventory_id)}">Delete</button>
                    </div>
                </td>
            </tr>
        `;
    }).join("");
}

function renderInventoryAlerts(lowStock, expired, expiring) {
    const alertBox = byId("inventoryAlerts");
    const messages = [];

    lowStock.forEach(item => {
        messages.push(`
            <div class="inventory-alert warning">
                ⚠ Low stock: <strong>${escapeHTML(item.item_name)}</strong>
                has ${formatNumber(item.quantity)} ${escapeHTML(item.unit)} remaining.
            </div>
        `);
    });

    expired.forEach(item => {
        messages.push(`
            <div class="inventory-alert danger">
                ⛔ Expired: <strong>${escapeHTML(item.item_name)}</strong>
                passed its expiry date.
            </div>
        `);
    });

    expiring.forEach(item => {
        messages.push(`
            <div class="inventory-alert warning">
                ⏰ Expiring soon: <strong>${escapeHTML(item.item_name)}</strong>
                expires ${escapeHTML(formatDate(item.expiry_date))}.
            </div>
        `);
    });

    alertBox.innerHTML = messages.length
        ? messages.join("")
        : `<div class="inventory-alert warning">No low-stock or expiry alerts at this time.</div>`;
}

// ADD OR UPDATE INVENTORY

byId("inventoryForm").addEventListener("submit", async event => {
    event.preventDefault();

    const inventoryId = byId("inventoryId").value;
    const submitButton = byId("inventorySubmitBtn");
    const originalText = submitButton.textContent;

    const payload = {
        item_name: byId("itemName").value.trim(),
        food_category: byId("inventoryCategory").value,
        business_type: byId("inventoryBusiness").value,
        quantity: byId("inventoryQuantity").value,
        unit: byId("inventoryUnit").value,
        purchase_date: byId("purchaseDate").value,
        expiry_date: byId("expiryDate").value,
        reorder_level: byId("reorderLevel").value,
        storage_temperature: byId("inventoryTemperature").value || null
    };

    if (payload.expiry_date < payload.purchase_date) {
        showMessage("inventoryMessage",
            "Expiry date cannot be before purchase date.", "error");
        return;
    }

    submitButton.disabled = true;
    submitButton.textContent = "Saving...";

    try {
        let result;

        if (inventoryId) {
            result = await apiRequest(`/api/inventory/${inventoryId}`, {
                method: "PUT",
                body: JSON.stringify(payload)
            });
        } else {
            result = await apiRequest("/api/inventory", {
                method: "POST",
                body: JSON.stringify(payload)
            });
        }

        showMessage("inventoryMessage", result.message || "Saved successfully.");
        resetInventoryForm();
        await loadInventory();

    } catch (error) {
        showMessage("inventoryMessage", error.message, "error");
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = byId("inventoryId").value
            ? "Update Item" : originalText;
    }
});

// INVENTORY EDIT AND DELETE

byId("inventoryTableBody").addEventListener("click", async event => {
    const button = event.target.closest("button[data-action]");
    if (!button) return;

    const inventoryId = Number(button.dataset.id);
    const item = inventoryRecords.find(record =>
        Number(record.inventory_id) === inventoryId
    );

    if (!item) {
        showMessage("inventoryMessage", "Inventory item not found.", "error");
        return;
    }

    if (button.dataset.action === "edit") {
        editInventoryItem(item);
    }

    if (button.dataset.action === "delete") {
        await deleteInventoryItem(inventoryId, item.item_name);
    }
});

function editInventoryItem(item) {
    byId("inventoryId").value = item.inventory_id;
    byId("itemName").value = item.item_name;
    byId("inventoryCategory").value = item.food_category;
    byId("inventoryBusiness").value = item.business_type;
    byId("inventoryQuantity").value = item.quantity;
    byId("inventoryUnit").value = item.unit;
    byId("purchaseDate").value = formatDate(item.purchase_date);
    byId("expiryDate").value = formatDate(item.expiry_date);
    byId("reorderLevel").value = item.reorder_level;
    byId("inventoryTemperature").value = item.storage_temperature ?? "";

    byId("inventoryFormTitle").textContent = "Edit Inventory Item";
    byId("inventorySubmitBtn").textContent = "Update Item";
    byId("cancelEditBtn").classList.remove("hidden");

    byId("inventoryForm").scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}

async function deleteInventoryItem(inventoryId, itemName) {
    if (!confirm(`Are you sure you want to delete "${itemName}"?`)) return;

    try {
        const result = await apiRequest(`/api/inventory/${inventoryId}`, {
            method: "DELETE"
        });

        showMessage("inventoryMessage", result.message);
        await loadInventory();

    } catch (error) {
        showMessage("inventoryMessage", error.message, "error");
    }
}

function resetInventoryForm() {
    byId("inventoryForm").reset();
    byId("inventoryId").value = "";
    byId("reorderLevel").value = "5";

    byId("inventoryFormTitle").textContent = "Add Inventory Item";
    byId("inventorySubmitBtn").textContent = "Add Item";
    byId("cancelEditBtn").classList.add("hidden");

    byId("purchaseDate").value = localDateString();
}

byId("cancelEditBtn").addEventListener("click", resetInventoryForm);
byId("refreshInventoryBtn").addEventListener("click", loadInventory);

// =====================================================
// FOOD DONATION MANAGEMENT
// =====================================================

async function loadDonations() {
    const tableBody = byId("donationTableBody");

    tableBody.innerHTML =
        `<tr><td colspan="11" class="empty-state">Loading donations...</td></tr>`;

    try {
        const data = await apiRequest("/api/donations");
        donationRecords = data.donations || [];

        renderDonations(donationRecords);

    } catch (error) {
        console.error("Donation loading error:", error);

        tableBody.innerHTML =
            `<tr><td colspan="11" class="empty-state">Could not load donations: ${escapeHTML(error.message)}</td></tr>`;
    }
}

function donationStatusBadge(status) {
    const classes = {
        Pending: "badge-pending",
        Accepted: "badge-accepted",
        Completed: "badge-completed",
        Cancelled: "badge-cancelled"
    };

    const className = classes[status] || "badge-pending";

    return `<span class="badge ${className}">${escapeHTML(status)}</span>`;
}

function renderDonations(items) {
    byId("donationCount").textContent = items.length;

    byId("pendingDonationCount").textContent =
        items.filter(item => item.donation_status === "Pending").length;

    byId("acceptedDonationCount").textContent =
        items.filter(item => item.donation_status === "Accepted").length;

    byId("completedDonationCount").textContent =
        items.filter(item => item.donation_status === "Completed").length;

    const tableBody = byId("donationTableBody");

    if (items.length === 0) {
        tableBody.innerHTML =
            `<tr><td colspan="11" class="empty-state">No donation records yet. Register your first donation above.</td></tr>`;
        return;
    }

    tableBody.innerHTML = items.map(item => `
        <tr>
            <td>${escapeHTML(item.donation_id)}</td>
            <td><strong>${escapeHTML(item.food_name)}</strong></td>
            <td>${escapeHTML(item.food_category)}</td>
            <td>${formatNumber(item.quantity)} ${escapeHTML(item.unit)}</td>
            <td>${escapeHTML(item.donor_name)}</td>
            <td>${escapeHTML(item.recipient_name)}</td>
            <td>${escapeHTML(item.recipient_contact || "-")}</td>
            <td>${escapeHTML(formatDate(item.donation_date))}</td>
            <td>${escapeHTML(formatDate(item.expiry_date))}</td>
            <td>${donationStatusBadge(item.donation_status)}</td>
            <td>
                <div class="action-buttons">
                    <button class="btn btn-outline btn-small"
                        data-action="edit"
                        data-id="${Number(item.donation_id)}">Edit</button>
                    <button class="btn btn-danger btn-small"
                        data-action="delete"
                        data-id="${Number(item.donation_id)}">Delete</button>
                </div>
            </td>
        </tr>
    `).join("");
}

// ADD OR UPDATE DONATION

byId("donationForm").addEventListener("submit", async event => {
    event.preventDefault();

    const donationId = byId("donationId").value;
    const submitButton = byId("donationSubmitBtn");
    const originalText = submitButton.textContent;

    const payload = {
        food_name: byId("donationFoodName").value.trim(),
        food_category: byId("donationCategory").value,
        quantity: byId("donationQuantity").value,
        unit: byId("donationUnit").value,
        donor_name: byId("donorName").value.trim(),
        recipient_name: byId("recipientName").value.trim(),
        recipient_contact: byId("recipientContact").value.trim(),
        donation_date: byId("donationDate").value,
        expiry_date: byId("donationExpiryDate").value,
        donation_status: byId("donationStatus").value,
        notes: byId("donationNotes").value.trim()
    };

    if (payload.expiry_date < payload.donation_date) {
        showMessage(
            "donationMessage",
            "Food expiry date cannot be before donation date.",
            "error"
        );
        return;
    }

    submitButton.disabled = true;
    submitButton.textContent = "Saving...";

    try {
        let result;

        if (donationId) {
            result = await apiRequest(`/api/donations/${donationId}`, {
                method: "PUT",
                body: JSON.stringify(payload)
            });
        } else {
            result = await apiRequest("/api/donations", {
                method: "POST",
                body: JSON.stringify(payload)
            });
        }

        showMessage("donationMessage", result.message || "Donation saved successfully.");

        resetDonationForm();
        await loadDonations();

    } catch (error) {
        showMessage("donationMessage", error.message, "error");
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = byId("donationId").value
            ? "Update Donation" : originalText;
    }
});

// EDIT AND DELETE DONATIONS

byId("donationTableBody").addEventListener("click", async event => {
    const button = event.target.closest("button[data-action]");
    if (!button) return;

    const donationId = Number(button.dataset.id);

    const item = donationRecords.find(record =>
        Number(record.donation_id) === donationId
    );

    if (!item) {
        showMessage("donationMessage", "Donation record not found.", "error");
        return;
    }

    if (button.dataset.action === "edit") {
        editDonation(item);
    }

    if (button.dataset.action === "delete") {
        await deleteDonation(donationId, item.food_name);
    }
});

function editDonation(item) {
    byId("donationId").value = item.donation_id;
    byId("donationFoodName").value = item.food_name;
    byId("donationCategory").value = item.food_category;
    byId("donationQuantity").value = item.quantity;
    byId("donationUnit").value = item.unit;
    byId("donorName").value = item.donor_name;
    byId("recipientName").value = item.recipient_name;
    byId("recipientContact").value = item.recipient_contact || "";
    byId("donationDate").value = formatDate(item.donation_date);
    byId("donationExpiryDate").value = formatDate(item.expiry_date);
    byId("donationStatus").value = item.donation_status;
    byId("donationNotes").value = item.notes || "";

    byId("donationFormTitle").textContent = "Edit Food Donation";
    byId("donationSubmitBtn").textContent = "Update Donation";
    byId("cancelDonationEditBtn").classList.remove("hidden");

    byId("donationForm").scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}

async function deleteDonation(donationId, foodName) {
    if (!confirm(`Are you sure you want to delete the donation "${foodName}"?`)) {
        return;
    }

    try {
        const result = await apiRequest(`/api/donations/${donationId}`, {
            method: "DELETE"
        });

        showMessage("donationMessage", result.message);
        await loadDonations();

    } catch (error) {
        showMessage("donationMessage", error.message, "error");
    }
}

function resetDonationForm() {
    byId("donationForm").reset();
    byId("donationId").value = "";

    byId("donationFormTitle").textContent = "Register Food Donation";
    byId("donationSubmitBtn").textContent = "Register Donation";
    byId("donationStatus").value = "Pending";
    byId("cancelDonationEditBtn").classList.add("hidden");

    byId("donationDate").value = localDateString();
}

byId("cancelDonationEditBtn").addEventListener("click", resetDonationForm);
byId("refreshDonationsBtn").addEventListener("click", loadDonations);


// =====================================================
// DEMAND FORECASTING
// =====================================================

byId("demandForm").addEventListener("submit", async event => {

    event.preventDefault();

    const submitButton = byId("demandSubmitBtn");
    const originalText = submitButton.textContent;

    const payload = {
        center_id: byId("centerId").value,
        meal_id: byId("mealId").value,
        week: byId("forecastWeek").value,
        checkout_price: byId("checkoutPrice").value,
        base_price: byId("basePrice").value,
        emailer_for_promotion: byId("emailerPromotion").value,
        homepage_featured: byId("homepageFeatured").value
    };

    const resultBox = byId("demandResult");

    submitButton.disabled = true;
    submitButton.textContent = "Generating Forecast...";

    resultBox.classList.add("hidden");

    try {

        const data = await apiRequest("/api/predict-demand", {
            method: "POST",
            body: JSON.stringify(payload)
        });

        const prediction = data.prediction;
        const features = data.historical_features;

        resultBox.innerHTML = `

            <h3>Demand Forecast Generated Successfully</h3>

            <p>
                <strong>Predicted Orders:</strong>
                ${formatNumber(prediction.predicted_orders)}
                orders
            </p>

            <p>
                <strong>Fulfilment Centre:</strong>
                ${escapeHTML(data.input.center_id)}
            </p>

            <p>
                <strong>Meal ID:</strong>
                ${escapeHTML(data.input.meal_id)}
            </p>

            <p>
                <strong>Forecast Week:</strong>
                ${escapeHTML(data.input.week)}
            </p>

            <p class="notice">
                ${escapeHTML(data.notice)}
            </p>

        `;

        resultBox.classList.remove("hidden");

    } catch (error) {

        resultBox.textContent =
            `Demand prediction failed: ${error.message}`;

        resultBox.classList.remove("hidden");

    } finally {

        submitButton.disabled = false;
        submitButton.textContent = originalText;

    }

});


// =====================================================
// MOBILE NAVIGATION DRAWER
// =====================================================

(function () {
    function initializeMobileNavigation() {
        const appLayout = document.querySelector(".app-layout");
        const toggleButton = document.getElementById("mobileNavToggle");
        const backdrop = document.getElementById("mobileNavBackdrop");
        const sidebar = document.getElementById("appSidebar");

        // Check that all required HTML elements exist
        if (!appLayout || !toggleButton || !backdrop || !sidebar) {
            console.error(
                "Mobile navigation elements are missing. Check index.html for mobileNavToggle, mobileNavBackdrop and appSidebar."
            );
            return;
        }

        // Open the navigation menu
        function openMenu() {
            appLayout.classList.add("mobile-nav-open");

            toggleButton.setAttribute("aria-expanded", "true");
            toggleButton.setAttribute("aria-label", "Close navigation menu");

            backdrop.setAttribute("aria-hidden", "false");
        }

        // Close the navigation menu
        function closeMenu() {
            appLayout.classList.remove("mobile-nav-open");

            toggleButton.setAttribute("aria-expanded", "false");
            toggleButton.setAttribute("aria-label", "Open navigation menu");

            backdrop.setAttribute("aria-hidden", "true");
        }

        // Toggle menu when hamburger button is clicked
        toggleButton.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();

            const isOpen = appLayout.classList.contains("mobile-nav-open");

            if (isOpen) {
                closeMenu();
            } else {
                openMenu();
            }
        });

        // Close menu when the user clicks outside the sidebar
        backdrop.addEventListener("click", function () {
            closeMenu();
        });

        // Close menu after selecting a navigation item
        sidebar.querySelectorAll(".nav-link").forEach(function (link) {
            link.addEventListener("click", function () {
                closeMenu();
            });
        });

        // Close menu when Escape key is pressed
        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
                closeMenu();
            }
        });

        // Close menu when switching back to desktop screen size
        window.addEventListener("resize", function () {
            if (window.innerWidth > 760) {
                closeMenu();
            }
        });

        // Ensure menu starts closed on mobile
        if (window.innerWidth <= 760) {
            closeMenu();
        }

        console.log("FoodWise AI mobile navigation initialized successfully.");
    }

    // Initialize after HTML has loaded
    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initializeMobileNavigation,
            { once: true }
        );
    } else {
        initializeMobileNavigation();
    }
})();

// =====================================================
// REFRESH BUTTONS
// =====================================================

byId("refreshHistoryBtn").addEventListener("click", loadPredictionHistory);

byId("dashboardHistoryRefresh").addEventListener("click", async () => {
    await loadPredictionHistory();
    await loadDashboardStats();
});

// =====================================================
// INITIALIZE APPLICATION
// =====================================================

document.addEventListener("DOMContentLoaded", async () => {
    setDefaultDates();

    await checkBackend();

    await Promise.all([
        loadDashboardStats(),
        loadPredictionHistory()
    ]);
});