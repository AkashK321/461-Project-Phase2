const API_URL = "https://bbdudjvh08.execute-api.us-east-2.amazonaws.com"; 

let authToken = localStorage.getItem('auth_token');
let currentUser = null;

if (authToken) {
    const decoded = parseJwt(authToken);
    if (decoded) {
        currentUser = decoded;
    } else {
        // If token is invalid/expired, clear it
        authToken = null;
        localStorage.removeItem('auth_token');
    }
}

// --- Router Logic ---
window.addEventListener('load', router);
window.addEventListener('hashchange', router);

function router() {
    const hash = window.location.hash || '#/login';

    if (!authToken && hash !== '#/login') {
        window.location.hash = '#/login';
        return;
    }

    if (authToken && hash === '#/login') {
        window.location.hash = '#/dashboard';
        return;
    }

    // 3. Close all modals by default
    const allDialogs = document.querySelectorAll('dialog');
    allDialogs.forEach(d => d.close());

    // 4. Handle Routes
    if (hash === '#/login') {
        updateUI(false);
    } 
    else if (hash === '#/dashboard') {
        updateUI(true);
    } 
    else if (hash === '#/admin/create-user') {
        updateUI(true);
        prepCreateUserModal();
        document.getElementById('create-user-modal').showModal();
    } 
    else if (hash === '#/admin/view-users') {
        updateUI(true);
        fetchUsersList(); 
        document.getElementById('view-users-modal').showModal();
    }
    else if (hash === '#/admin/reset') {
            resetRegistry();
            window.location.hash = '#/dashboard';
    }
}

// --- Auth & JWT ---
function parseJwt (token) {
    try {
        var base64Url = token.split('.')[1];
        var base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        var jsonPayload = decodeURIComponent(window.atob(base64).split('').map(function(c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
        return JSON.parse(jsonPayload);
    } catch (e) {
        return null;
    }
}

async function handleLogin() {
    const name = document.getElementById('username').value;
    const pass = document.getElementById('password').value;
    const errDiv = document.getElementById('login-error');
    
    if (!name || !pass) {
        errDiv.innerText = "Please enter both username and password.";
        errDiv.classList.remove('hidden');
        return;
    }

    try {
        const res = await fetch(`${API_URL}/authenticate`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ user: { name: name }, secret: { password: pass } })
        });

        if(res.ok) {
            const txt = await res.json();
            // Extract token
            const rawToken = txt.replace(/"/g, '').split(' ')[1]; 
            
            const decoded = parseJwt(rawToken);
            if(decoded) {
                authToken = rawToken;
                currentUser = decoded;
                
                localStorage.setItem('auth_token', authToken);

                errDiv.classList.add('hidden');
                window.location.hash = '#/dashboard';
            } else {
                errDiv.innerText = "System Error: Failed to parse authentication token.";
                errDiv.classList.remove('hidden');
            }
        } else {
            errDiv.innerText = "Invalid credentials. Please try again.";
            errDiv.classList.remove('hidden');
        }
    } catch(e) {
        errDiv.innerText = "Connection failed. Please check your internet connection.";
        errDiv.classList.remove('hidden');
    }
}

function handleLogout() {
    authToken = null;
    currentUser = null;
    
    localStorage.removeItem('auth_token');

    document.getElementById('username').value = '';
    document.getElementById('password').value = '';
    document.getElementById('login-error').classList.add('hidden');
    
    document.getElementById('upload-res').innerHTML = '';
    document.getElementById('search-res').innerHTML = '';

    window.location.hash = '#/login';
}

function updateUI(isLoggedIn) {
    const nav = document.getElementById('nav-bar');
    const login = document.getElementById('login-section');
    const dash = document.getElementById('dashboard');
    
    if(isLoggedIn) {
        nav.classList.remove('hidden');
        login.classList.add('hidden');
        dash.classList.remove('hidden');

        const isAdmin = currentUser && currentUser.roles && currentUser.roles.includes('admin');
        
        document.querySelectorAll('.admin-only').forEach(el => {
            if(isAdmin) el.classList.remove('hidden');
            else el.classList.add('hidden');
        });

        document.querySelectorAll('.user-only').forEach(el => {
            if(!isAdmin) el.classList.remove('hidden');
            else el.classList.add('hidden');
        });

    } else {
        nav.classList.add('hidden');
        login.classList.remove('hidden');
        dash.classList.add('hidden');
    }
}

// --- RESULT RENDERING HELPERS ---
function formatKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function extractScores(item, excludedKeys, scoreKeywords) {
    const foundScores = {};
    if (item.scores && typeof item.scores === 'object' && !Array.isArray(item.scores)) {
        for (const [sKey, sVal] of Object.entries(item.scores)) {
            foundScores[sKey] = sVal;
        }
    }
    for (const [key, value] of Object.entries(item)) {
        if (excludedKeys.includes(key) || key === 'scores') continue;
        if (scoreKeywords.some(kw => key.toLowerCase().includes(kw))) {
            foundScores[key] = value;
        }
    }
    return foundScores;
}

async function downloadArtifact(type, id) {
    const artifactType = type || 'model'; 
    try {
        const res = await fetch(`${API_URL}/artifacts/${artifactType}/${id}`, {
            method: 'GET',
            headers: { 'X-Authorization': `bearer ${authToken}` }
        });
        if (!res.ok) throw new Error("Failed to fetch download URL");
        
        const json = await res.json();
        const downloadUrl = json.data && json.data.download_url;

        if (downloadUrl) {
            window.open(downloadUrl, '_blank');
        } else {
            alert("No download URL available for this artifact.");
        }
    } catch (err) { alert("Download failed: " + err.message); }
}

function renderDetailView(item) {
    const excludedKeys = ['model_name', 'Name', 'version', 'id']; 
    const scoreKeywords = [
        'net_score', 'ramp_up', 'correctness', 'bus_factor', 
        'responsive_maintainer', 'license', 'dependency_pinning', 
        'code_review', 'code_quality', 'dataset_quality', 
        'performance_claims', 'dataset_and_code', 'size',
        'netscore'
    ];

    const scoresObj = extractScores(item, excludedKeys, scoreKeywords);
    const scoreKeys = Object.keys(scoresObj);

    let generalHtml = '';
    for (const [key, value] of Object.entries(item)) {
        if (excludedKeys.includes(key)) continue;
        if (key === 'scores') continue; 
        if (scoreKeys.includes(key)) continue; 

        let displayValue = value;
        
        if (key === 's3_key') {
            const safeType = item.type || 'model';
            displayValue = `<a href="#" class="download-link" onclick="downloadArtifact('${safeType}', '${item.id}'); return false;">${value} ⬇</a>`;
        }
        else if (typeof value === 'object' && value !== null) {
            displayValue = JSON.stringify(value); 
        } 
        else if (String(value).startsWith('http')) {
            displayValue = `<a href="${value}" target="_blank" rel="noopener noreferrer">Link</a>`;
        }

        generalHtml += `
            <div class="detail-item">
                <div class="detail-label">${formatKey(key)}</div>
                <div class="detail-value">${displayValue}</div>
            </div>
        `;
    }

    let scoresRows = '';
    if (scoreKeys.length > 0) {
        scoreKeys.sort().forEach(key => {
            let val = scoresObj[key];
            let displayVal = val;
            if (typeof val === 'object' && val !== null) {
                    const subEntries = Object.entries(val).map(([k, v]) => {
                    return `<li><span style="color:#555; font-weight:500;">${formatKey(k)}:</span> ${v}</li>`;
                }).join('');
                displayVal = `<ul style="margin:0; padding-left:1.2rem; font-size:0.85rem;">${subEntries}</ul>`;
            }
            scoresRows += `<tr><td style="width: 40%;"><strong>${formatKey(key)}</strong></td><td>${displayVal}</td></tr>`;
        });
    }

    let output = '';
    if (generalHtml) output += `<div class="detail-grid">${generalHtml}</div>`;
    else output += '<p style="color:#666; font-style:italic;">No general details available.</p>';

    if (scoresRows) {
        output += `
            <details class="scores-dropdown">
                <summary>Quality Scores</summary>
                <table class="scores-table">
                    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                    <tbody>${scoresRows}</tbody>
                </table>
            </details>
        `;
    }
    return output;
}

function renderUploadResult(data) {
    if (data.metadata) {
        const meta = data.metadata;
        return `
            <div class="success-card">
                <h4>Upload Successful</h4>
                <p><strong>Name:</strong> ${meta.name || 'N/A'}</p>
                <p><strong>Version:</strong> ${meta.version || '1.0.0'}</p>
                <p><strong>ID:</strong> ${meta.id}</p>
            </div>
        `;
    } 
    return `<div style="background:#333; color:#fff; padding:1rem; border-radius:5px;"><pre style="margin:0; color:#fff;">${JSON.stringify(data, null, 2)}</pre></div>`;
}

function renderSearchResult(data) {
    if (Array.isArray(data)) {
        if (data.length === 0) return `<p>No packages found matching your query.</p>`;

        const rows = data.map((item, index) => {
            const detailId = `detail-${index}`;
            const formattedDetails = renderDetailView(item);
            
            const displayName = item.name || item.Name || item.model_name || (item.metadata && item.metadata.name) || 'Unknown';
            const displayType = item.type || 'Unknown';

            return `
                <tr class="main-row" 
                    id="row-${index}"
                    onclick="toggleRow('${detailId}', 'row-${index}')" 
                    onkeydown="handleRowKey(event, '${detailId}', 'row-${index}')"
                    tabindex="0" 
                    role="button" 
                    aria-expanded="false" 
                    aria-controls="${detailId}">
                    
                    <td><span class="expand-icon">▶</span> ${displayName}</td>
                    <td>${displayType}</td>
                    <td><code>${item.id}</code></td>
                </tr>
                <tr id="${detailId}" class="detail-row hidden">
                    <td colspan="3">
                        <div class="detail-content">${formattedDetails}</div>
                    </td>
                </tr>
            `;
        }).join('');

        return `
            <table class="table-container" role="grid" aria-label="Search Results">
                <thead>
                    <tr style="background:#f8f9fa;">
                        <th>Name</th>
                        <th>Type</th>
                        <th>ID</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
            <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">Tip: Click a row to see full details.</p>
        `;
    }
    return `<div style="background:#333; color:#fff; padding:1rem; border-radius:5px;"><pre style="margin:0; color:#fff;">${JSON.stringify(data, null, 2)}</pre></div>`;
}

function toggleRow(detailId, mainRowId) {
    const detailRow = document.getElementById(detailId);
    const mainRow = document.getElementById(mainRowId);
    
    const isHidden = detailRow.classList.contains('hidden');
    if (isHidden) {
        detailRow.classList.remove('hidden');
        mainRow.classList.add('expanded');
        mainRow.setAttribute('aria-expanded', 'true');
    } else {
        detailRow.classList.add('hidden');
        mainRow.classList.remove('expanded');
        mainRow.setAttribute('aria-expanded', 'false');
    }
}

function handleRowKey(event, detailId, mainRowId) {
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        toggleRow(detailId, mainRowId);
    }
}

// --- Admin User Management Logic ---

function prepCreateUserModal() {
    document.getElementById('new-user-name').value = '';
    document.getElementById('new-user-pass').value = '';
    document.getElementById('role-admin').checked = false;
    document.querySelectorAll('.role-chk').forEach(c => {
        c.checked = false; 
        c.disabled = false;
    });
}

function toggleAdminRoles(adminCheckbox) {
    const others = document.querySelectorAll('.role-chk');
    others.forEach(chk => {
        if(adminCheckbox.checked) {
            chk.checked = true;
            chk.disabled = true;
        } else {
            chk.disabled = false;
            chk.checked = false;
        }
    });
}

async function createUser() {
    const name = document.getElementById('new-user-name').value;
    const pass = document.getElementById('new-user-pass').value;
    const isAdmin = document.getElementById('role-admin').checked;
    
    let roles = [];
    if(isAdmin) {
        roles = ['admin', 'upload', 'search', 'download'];
    } else {
        document.querySelectorAll('.role-chk').forEach(c => {
            if(c.checked) roles.push(c.value);
        });
    }

    if(!name || !pass || roles.length === 0) {
        alert("Please provide a username, password, and select at least one role.");
        return;
    }

    try {
        const res = await fetch(`${API_URL}/users`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Authorization': `bearer ${authToken}` },
            body: JSON.stringify({ username: name, password: pass, roles: roles })
        });
        if(res.ok) {
            alert("User Created Successfully!");
            window.location.hash = '#/dashboard';
        } else {
            const json = await res.json();
            alert("Error: " + (json.error || "Unknown"));
        }
    } catch(e) { alert("Network Error: " + e.message); }
}

async function fetchUsersList() {
    const list = document.getElementById('users-list-container');
    list.innerHTML = "<p>Loading users...</p>";

    try {
        const res = await fetch(`${API_URL}/users`, {
            headers: { 'X-Authorization': `bearer ${authToken}` }
        });
        const users = await res.json();
        
        if(!res.ok) throw new Error(users.error || "Failed to fetch");

        let html = `<table aria-label="List of users">
            <thead><tr><th scope="col">Username</th><th scope="col">Roles</th><th scope="col">Actions</th></tr></thead>
            <tbody>`;
        
        users.forEach(u => {
            html += `<tr>
                <td><strong>${u.username}</strong></td>
                <td>${u.roles.join(', ')}</td>
                <td>
                    <button onclick="openModifyRoles('${u.id}', '${u.username}', '${u.roles.join(',')}')" 
                        aria-label="Edit roles for ${u.username}"
                        style="font-size:0.9em; padding:6px 12px; width:auto; display:inline-block; margin-right:5px;">
                        Edit
                    </button>
                    <button onclick="deleteUser('${u.id}', '${u.username}')" 
                        aria-label="Delete user ${u.username}"
                        style="font-size:0.9em; padding:6px 12px; width:auto; display:inline-block; background:var(--danger-color);">
                        Delete
                    </button>
                </td>
            </tr>`;
        });
        html += "</tbody></table>";
        list.innerHTML = html;
    } catch(e) {
        list.innerHTML = `<div class="error-box">Error loading users: ${e.message}</div>`;
    }
}

function openModifyRoles(id, name, rolesStr) {
    document.getElementById('mod-user-id').value = id;
    document.getElementById('mod-user-display').innerText = "Editing User: " + name;
    
    const roles = rolesStr.split(',');
    const isAdmin = roles.includes('admin');

    document.getElementById('mod-role-admin').checked = isAdmin;
    document.querySelectorAll('.mod-chk').forEach(c => {
        c.checked = roles.includes(c.value) || isAdmin;
    });
    document.getElementById('modify-roles-modal').showModal();
}

async function submitRoleUpdate() {
    const id = document.getElementById('mod-user-id').value;
    const isAdmin = document.getElementById('mod-role-admin').checked;
    let roles = [];
    
    if(isAdmin) {
        roles = ['admin', 'upload', 'search', 'download'];
    } else {
        document.querySelectorAll('.mod-chk').forEach(c => {
            if(c.checked) roles.push(c.value);
        });
    }

    try {
        const res = await fetch(`${API_URL}/users/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', 'X-Authorization': `bearer ${authToken}` },
            body: JSON.stringify({ roles: roles })
        });
        if(res.ok) {
            alert("Roles Updated Successfully");
            document.getElementById('modify-roles-modal').close();
            fetchUsersList(); 
        } else {
            const json = await res.json();
            alert("Error: " + json.error);
        }
    } catch(e) { alert("Error: " + e.message); }
}

async function deleteUser(id, username) {
    if(!confirm(`Are you sure you want to delete user "${username}"?`)) return;
    try {
        const res = await fetch(`${API_URL}/users/${id}`, {
            method: 'DELETE',
            headers: { 'X-Authorization': `bearer ${authToken}` }
        });
        if(res.ok) {
            fetchUsersList();
        } else {
            const json = await res.json();
            alert("Error: " + json.error);
        }
    } catch(e) { alert("Error: " + e.message); }
}

async function deleteSelf() {
    if(!confirm("Are you sure you want to delete your account? This action cannot be undone.")) return;
    try {
        const res = await fetch(`${API_URL}/users/${currentUser.sub}`, {
            method: 'DELETE',
            headers: { 'X-Authorization': `bearer ${authToken}` }
        });
        if(res.ok) {
            alert("Account deleted.");
            handleLogout();
        } else {
            const json = await res.json();
            alert("Error: " + json.error);
        }
    } catch(e) { alert("Error: " + e.message); }
}

async function uploadPackage() {
    const url = document.getElementById('pkg-url').value;
    const resDiv = document.getElementById('upload-res');
    
    if(!url) {
        resDiv.innerHTML = "<div class='error-box'>Please enter a URL.</div>";
        return;
    }

    resDiv.innerHTML = "<p>Uploading...</p>";
    try {
        const res = await fetch(`${API_URL}/artifact/model`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'X-Authorization': `bearer ${authToken}`},
            body: JSON.stringify({ url: url })
        });
        const json = await res.json();
        
        resDiv.innerHTML = renderUploadResult(json);
        
    } catch(e) { resDiv.innerHTML = `<div class='error-box'>Error: ${e.message}</div>`; }
}

async function searchPackages() {
    const q = document.getElementById('search-q').value;
    const resDiv = document.getElementById('search-res');
    resDiv.innerHTML = "<p>Searching...</p>";
    try {
        const res = await fetch(`${API_URL}/artifacts`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'X-Authorization': `bearer ${authToken}`},
            body: JSON.stringify([{ name: q }])
        });
        const json = await res.json();
        
        resDiv.innerHTML = renderSearchResult(json);
        
    } catch(e) { resDiv.innerHTML = `<div class='error-box'>Error: ${e.message}</div>`; }
}

async function resetRegistry() {
    if(!confirm("WARNING: This will wipe the entire registry database. Continue?")) return;
    try {
        const res = await fetch(`${API_URL}/reset`, {
            method: 'DELETE',
            headers: {'X-Authorization': `bearer ${authToken}`}
        });
        const json = await res.json();
        alert(JSON.stringify(json));
    } catch(e) { alert("Error: " + e.message); }
}