import {app} from "../../../scripts/app.js";
import {api} from "../../../scripts/api.js";

export function setupButtons(menu, app, api) {
    // Save To Git Button
    const saveButton = document.createElement("button");
    saveButton.textContent = "Save To Git";
    saveButton.onclick = async () => {
        try {
            const json = JSON.stringify(app.graph.serialize(), null, 2);
            const userNameValue = menu.querySelector("#branch-name-field").value;
            const dropDownSelected = menu.querySelector("#graph-name-field").value;

            await api.fetchApi("/gde/git/save", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    data: json,
                    file_name: dropDownSelected,
                    user_name: userNameValue
                }),
            });
        } catch (error) {
            console.error(error);
        }
    };
    menu.append(saveButton);

    // Load From Git Button
    const loadButton = document.createElement("button");
    loadButton.textContent = "Load From Git";
    loadButton.onclick = async () => {
        try {
            const response = await api.fetchApi("/gde/git/load", {
                // needs user_name file_name
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                        user_name: menu.querySelector("#branch-name-field").value,
                        file_name: menu.querySelector("#graph-name-field").value
                    }
                ),

            });
            const data = await response.json();
            app.loadGraphData(JSON.parse(data.data));
        } catch (error) {
            console.error(error);
        }
    };
    menu.append(loadButton);
}


export function setupModal() {
    // Modal
    const modal = document.createElement("div");
    modal.style.display = "none"; // Initially hidden
    modal.style.position = "fixed";
    modal.style.top = "50%";
    modal.style.left = "50%";
    modal.style.transform = "translate(-50%, -50%)";
    modal.style.padding = "20px";
    modal.style.backgroundColor = "#fff";
    modal.style.border = "1px solid #000";
    modal.style.zIndex = "1000"; // Ensure it's on top of other elements
    document.body.append(modal);

    // Input field inside the modal
    const modalInput = document.createElement("input");
    modalInput.type = "text";
    modalInput.placeholder = "Enter graph name";
    modal.append(modalInput);

    // Submit button inside the modal
    const submitButton = document.createElement("button");
    submitButton.textContent = "Submit";
    let targetDropdownId = ''; // This will store the ID of the dropdown we're targeting
    submitButton.onclick = () => {
        const dropdown = document.querySelector(`#${targetDropdownId}`);
        const newOption = document.createElement("option");
        newOption.value = modalInput.value;
        newOption.textContent = modalInput.value;
        dropdown.append(newOption);
        dropdown.value = modalInput.value;
        console.log("Submitted value:", modalInput.value);
        modal.style.display = "none"; // Hide the modal
    };
    modal.append(submitButton);

    // Function to show the modal (this can be used outside as well)
    window.showModal = function (dropdownId) {
        targetDropdownId = dropdownId; // Store the passed dropdown ID
        modal.style.display = "block";
    };
}


export function setupDropdown(menu, api, placeholder, apiEndpoint, dropdownId, user = null, graph_name = null, onCreateNew = null) {
    let dropdown = document.getElementById(dropdownId);

    // If dropdown already exists, clear its options first
    if (dropdown) {
        dropdown.innerHTML = '';
    } else {
        // Create new dropdown if it doesn't exist
        dropdown = document.createElement("select");
        dropdown.style.marginTop = "2px";
        dropdown.style.width = "100%";
        dropdown.style.borderRadius = "8px";
        dropdown.style.fontSize = "18px";
        dropdown.style.backgroundColor = "#222222";
        dropdown.style.color = "#EEEEEE";

        dropdown.id = dropdownId;
        //use comfy-list style

        menu.append(dropdown);
    }

    // Add default option
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = placeholder;
    dropdown.append(defaultOption);

    // Add "Create New" option
    const createNewOption = document.createElement("option");
    createNewOption.value = "create_new";
    createNewOption.textContent = "Create New";
    dropdown.append(createNewOption);

    dropdown.addEventListener("change", (event) => {
        // Check if "Create New" option was selected
        if (event.target.value === "create_new" && onCreateNew) {
            onCreateNew(event, dropdown);
        }

        // Check if branch dropdown is the one being changed
        if (dropdownId === 'branch-name-field') {
            while (dropdown.options.length > 2) {
                dropdown.remove(2);
            }
            setTimeout(async () => {
                await someFunction(dropdownId, api, apiEndpoint, placeholder, user, graph_name)
            })
        }
    });

    setTimeout(async () => {
        await someFunction(dropdownId, api, apiEndpoint, placeholder, user, graph_name)
    })
}

export async function someFunction(dropdownId, api, apiEndpoint, placeholder, user = null, graph_name = null) {
    try {
        const dropdown = document.getElementById(dropdownId);
        dropdown.disabled = true;  // Disable the file dropdown

        const queryParams = {
            user: user || ""
        };
        if (graph_name) {
            queryParams.graph_name = graph_name;
        }

        const response = await api.fetchApi(apiEndpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(queryParams)
        });
        const optionsData = await response.json();
        optionsData.forEach(optionItem => {
            const optionElement = document.createElement("option");
            optionElement.value = optionItem.value;
            optionElement.textContent = optionItem.label;
            dropdown.append(optionElement);
        });

        dropdown.disabled = false;  // Enable the file dropdown

    } catch (error) {
        console.error(`Error fetching options for ${placeholder}:`, error);
    }
}

