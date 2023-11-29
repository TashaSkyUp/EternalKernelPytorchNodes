import {app} from "../../../scripts/app.js";
import {api} from "../../../scripts/api.js";

export function setupButtons(gdeState, app, api) {
    // Save To Git Button
    const menu = document.querySelector(".comfy-menu");
    const saveButton = document.createElement("button");
    saveButton.textContent = "Save To Cloud";
    saveButton.onclick = async () => {
        try {
            const json = JSON.stringify(app.graph.serialize(), null, 2);
            const userNameValue = gdeState.user;
            const graphSelected = gdeState.graph;

            await api.fetchApi("/gde/git/save", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    data: json,
                    file_name: graphSelected,
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
    loadButton.textContent = "Load From Cloud";
    loadButton.onclick = async () => {
        try {
            const userNameValue = gdeState.user;
            const graphSelected = gdeState.graph;
            const response = await api.fetchApi("/gde/git/load", {
                // needs user_name file_name
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                        user_name: userNameValue,
                        file_name: graphSelected
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


export function setupModal(gdeState) {
    // Modal
    const modal = document.createElement("div");
    modal.id = "graph-name-modal";
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

    modal.submit = () => {
        // this is the function that will be called when the submit button is clicked
        // it will add the new option to the dropdown and select it
        const dropdown = document.querySelector(`#${targetDropdownId}`);
        const newOption = document.createElement("option");
        newOption.value = modalInput.value;
        newOption.textContent = modalInput.value;
        dropdown.append(newOption);
        dropdown.value = modalInput.value;
        console.log("Submitted value:", modalInput.value);
        modal.style.display = "none"; // Hide the modal
        return modalInput.value;
    }
    submitButton.onclick = () => {
        modal.submit()
    };
    modal.append(submitButton);

    // Function to show the modal (this can be used outside as well)
    window.showModal = function (dropdownId) {
        targetDropdownId = dropdownId; // Store the passed dropdown ID
        modal.style.display = "block";
    };
}

HTMLSelectElement.prototype.clearDropDown = function (placeHolder = "") {
    // remove everything
    while (this.options.length > 0) {
        this.remove(0);
    }
    // only add if there is a placeholder
    if (placeHolder != "") {
        //add select+type
        const defaultOption = document.createElement("option");
        defaultOption.value = "";
        defaultOption.textContent = placeHolder
        this.append(defaultOption);

        // Add "Create New" option
        const createNewOption = document.createElement("option");
        createNewOption.value = "create_new";
        createNewOption.textContent = "Create New";
        this.append(createNewOption);
    }

};


export function setupDropdown(menu, api, placeholder, apiEndpoint, dropdownId,
                              user = null, onCreateNew = null,
                              onChanged = null) {
    let dropdown = document.getElementById(dropdownId);

    // If dropdown already exists, clear its options first
    if (dropdown) {
        dropdown.clearDropDown(placeholder)
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
        dropdown.clearDropDown(placeholder)
        menu.append(dropdown);

        // Add event listener to the dropdown
        dropdown.addEventListener("change", (event) => {
            // Check if "Create New" option was selected
            if (event.target.value === "create_new" && onCreateNew) {
                onCreateNew(event, dropdown);
            }
            if (onChanged) {
                onChanged(event, dropdown);
            }

        }
        );
    }


    // dropdown.addEventListener("change", (event) => {
    //     // Check if "Create New" option was selected
    //     if (event.target.value === "create_new" && onCreateNew) {
    //         onCreateNew(event, dropdown);
    //     }
    //
    // });

    setTimeout(async () => {
        dropdown.clearDropDown(placeholder)
        await someFunction(dropdownId, api, apiEndpoint, placeholder, user)
    }, 50)
    return dropdown;
}

export async function someFunction(dropdownId, api, apiEndpoint, placeholder, user = null) {
    // this is the function that will be called to fetch the options for the dropdown
    try {
        const dropdown = document.getElementById(dropdownId);
        dropdown.disabled = true;  // Disable the file dropdown
        const clientId = api.socket.url.split("=")[1]
        const queryParams = {
            user: user || "",
            client_id: clientId
        };

        const response = await api.fetchApi(apiEndpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(queryParams)
        });
        const optionsData = await response.json();
        optionsData.forEach(optionItem => {
            // if the option already exists, don't add it again
            if (dropdown.querySelector(`option[value="${optionItem.value}"]`)) {
                return;
            }
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

