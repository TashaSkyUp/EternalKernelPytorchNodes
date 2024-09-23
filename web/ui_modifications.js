import {app} from "../../../scripts/app.js";
import {api} from "../../../scripts/api.js";
import {setupButtons, setupModal, setupDropdown} from './ui_helpers.js';


let loggedInEvent = null;
let loggedOutEvent = new Event('loggedOutEvent');
export let selectedGraph = null;

//const graphCreateNewEvent = new Event('graphCreateNew');
// create a place to store some variables, including security_token

class GDELogin {
    constructor(api) {
        this.api = api;
        this.security_token = null;
        this.initElements();
    }

    initElements() {
        this.login_form = this.createElement("form", {id: "gde-login"});
        // make sure the form doesnt cause the page to reload
        this.login_form.addEventListener("submit", (event) => {
            event.preventDefault();
        });
        //this.login_div = this.createElement("div", {id: "gde-login"});
        this.usernameField = this.createInputField("username-field", "Username");
        this.passwordField = this.createInputField("password-field", "Password", "password");
        this.loginButton = this.createButton("login-button", "Login", this.handleLogin.bind(this));
        this.logoutButton = this.createButton("logout-button", "Logout", this.handleLogout.bind(this));
    }

    createInputField(id, placeholder, type = "text") {
        return this.createElement("input", {
            id: id,
            placeholder: placeholder,
            type: type,
        });
    }

    createButton(id, text, clickHandler) {
        const button = this.createElement("button", {
            id: id,
            innerText: text,
        });
        button.addEventListener("click", clickHandler);
        return button;
    }

    createElement(tag, attributes) {
        const element = document.createElement(tag);
        for (let attr in attributes) {
            element[attr] = attributes[attr];
        }
        return element;
    }

    async handleLogin() {
        const username = this.usernameField.value;
        const password = this.passwordField.value;

        gdeState.user = username;

        const res = await this.api.fetchApi("/gde/login", {
            method: "POST",
            body: JSON.stringify({
                username: username,
                password: password,
                client_id: this.api.clientId,
            }),
        });

        if (res.status !== 200) {
            //if res.logged_in is false then the login failed but only need to alert if this.api.clientId is set
            if (res.logged_in === false && this.api.clientId) {
                alert("Login failed. Please check your username and password and try again.");
            }
            if (res.logged_in === true) {
                this.username = res.user;
                gdeState.user = username;
            }

        } else {
            const security_json = await res.json();
            this.security_token = security_json.security;
            this.switchToLogoutView();

            if (username) {
                const event = new CustomEvent('loggedInEvent',
                    {detail: {'user': username}});
                //loggedInEvent.target = {value:json.user};
                document.dispatchEvent(event);
            } else { // client was refreshed user was lost but client_id was not
                this.usernameField.value = security_json.user;
                loggedInEvent.target.value = security_json.user;
                document.dispatchEvent(loggedInEvent);

            }
            const welcomeMessage = this.createElement("div", {
                id: "welcome-message",
                innerText: `Welcome ${this.usernameField.value}`
            });
            this.login_form.append(welcomeMessage);
        }


    }

    async handleLogout() {
        await this.api.fetchApi("/gde/logout", {
            method: "POST",
            body: JSON.stringify({
                client_id: this.api.clientId,
            }),
        });
        document.dispatchEvent(loggedOutEvent);
        this.clearDropdowns();
        this.switchToLoginView();

        const welcomeMessage = document.getElementById("welcome-message");
        if (welcomeMessage) welcomeMessage.remove();

    }

    switchToLogoutView() {
        this.usernameField.remove();
        this.passwordField.remove();
        this.loginButton.remove();
        this.login_form.append(this.logoutButton);
    }

    switchToLoginView() {
        this.logoutButton.remove();
        this.login_form.append(this.usernameField, this.passwordField, this.loginButton);
    }

    clearDropdowns() {
        this.clearDropdown("graph-name-field");
        //this.clearDropdown("branch-name-field");
    }

    clearDropdown(id) {
        const dropdown = document.getElementById(id);
        if (dropdown) {
            while (dropdown.options.length > 2) {
                dropdown.remove(2);
            }
        }
    }


    async getServerState(client_id) {
        // check with the server to see if we have a valid client_id
        const response = await api.fetchApi("/gde/login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                client_id: client_id,
            }),
        });

        const data = await response.json();
        return data;
    }

    async setupLoginForm() {
        const menu = document.querySelector(".comfy-menu");
        // create a div in the menu for this extension

        //need to check with server for the status of our current api.clientId
        const cid = api.clientId || api.socket.url.split("=")[1];
        const server_state = await this.getServerState(cid).then((server_state) => {
            const logged_in = server_state.logged_in;
            gdeState.user = server_state.user;

            const separator = document.createElement("hr");
            separator.style.margin = "20px 0";
            separator.style.width = "100%";
            menu.append(separator);
            menu.append(this.login_form);

            if (logged_in) {
                this.login_form.append(this.logoutButton);

            } else {
                this.login_form.append(this.usernameField);
                this.login_form.append(this.passwordField);
                this.login_form.append(this.loginButton);

            }
        });
    }
}

const gdeLogin = new GDELogin(api);
let gdeState = {
    // this is for holding the state of the ui like what is selected etc
    graph: null,
    branch: null,
    user: null,
}

// create a dummy function
// gde.setup_login_form = async () => {
//     const menu = document.querySelector(".comfy-menu");
//     // create a div in the menu for this extension
//
//
//     //need to check with server for the status of our current api.clientId
//     const cid = api.clientId || api.socket.url.split("=")[1];
//     const server_state = await gde.get_server_state(cid).then((server_state) => {
//         const logged_in = server_state.logged_in;
//
//         const separator = document.createElement("hr");
//         separator.style.margin = "20px 0";
//         separator.style.width = "100%";
//         menu.append(separator);
//         menu.append(gde.login_div);
//
//         if (logged_in) {
//             gde.login_div.append(gde.logoutButton);
//
//         } else {
//             gde.login_div.append(gde.usernameField);
//             gde.login_div.append(gde.passwordField);
//             gde.login_div.append(gde.loginButton);
//
//         }
//     });
//
// }
// gde.hijack_queueprompt = async () => {
//     // is this actually needed?
// }
app.registerExtension({
    name: "GDE.UI.RemoveDefaultButtons",
    async setup(app) {
        document.getElementById("comfy-load-default-button").remove();
        document.getElementById("comfy-clipspace-button").remove();
    },
});

app.registerExtension({
    name: "GDE.UI.Login",
    async setup() {
        setTimeout(async () => {
            await gdeLogin.setupLoginForm();
            //await gde.hijack_queueprompt();
        }, 10);
    }


});

function SetupGraphDropDown(graphDropdown,
                            event,
                            menu,
                            handleGraphCreateNew,
                            handleGraphSelect) {
    // event may come in as null, so we need to populate it with default values
    if (!event) {
        event = {
            detail: {
                user: "main",
                graph: "default.json",
                branch: "main"
            },
            target: {
                value: "main"
            }
        }
        gdeState.user = "main"
    }
    // graphDropdown.clearDropDown();
    graphDropdown.clearDropDown("Select Graph");
    const selectedGraph = null
    graphDropdown = setupDropdown(menu, api,
        "Select Graph",
        "/gde/git/user_graphs",
        "graph-name-field",
        gdeState.user,
        handleGraphCreateNew,
        handleGraphSelect);
    // modify graphDropdown to change gdeState.graph when it is changed
    graphDropdown.addEventListener("change", (event) => {
        const selectedGraph = event.target.value;
        if (selectedGraph && selectedGraph !== gdeState.graph) {
            gdeState.graph = selectedGraph;
            const graphChangeEvent = new CustomEvent('graphChangeEvent', {
                detail: {
                    graph: selectedGraph
                }
            });
            document.dispatchEvent(graphChangeEvent);
        }
    });
    return graphDropdown;
}

app.registerExtension({
    name: "GDE.UI.Git",
    async setup() {
        setupModal();
        setupButtons(gdeState, app, api);


        const separator = document.createElement("hr");
        const menu = document.querySelector(".comfy-menu");
        separator.style.margin = "20px 0";
        separator.style.width = "100%";
        menu.append(separator);


        // const handleBranchCreateNew = (event, dropdown) => {
        //     // Custom logic for when "Create New" is selected in the branch dropdown
        //     showModal('branch-name-field'); // Show the modal for creating a new branch
        //     dropdown.value = ""; // Reset the dropdown value
        //     // Clear the file dropdown
        //     const fileDropdown = document.getElementById("graph-name-field");
        //
        //
        // };

        document.addEventListener('loggedInEvent', async (event) => {
            gdeState.user = event.detail.user;
            //gdeState.graph = event.detail.graph;
            graphDropdown = SetupGraphDropDown(
                graphDropdown,
                event,
                menu,
                handleGraphCreateNew,
                handleGraphSelect);

        });

        // no lets call clear dropdown when there is a loggout
        document.addEventListener('loggedOutEvent', async (event) => {
                graphDropdown.clearDropDown();
            }
        );

        const handleGraphCreateNew = (event, dropdown) => {
            // Custom logic for when "Create New" is selected in the branch dropdown
            showModal('graph-name-field');
            dropdown.disabled = true;

            const modal = document.getElementById("graph-name-modal");
            // Hijack its onsubmit function
            const prev_submit = modal.submit;

            modal.submit = () => {
                let graphName = prev_submit();
                if (graphName) {
                    // Add the new graph to the dropdown
                    const optionElement = document.createElement("option");
                    optionElement.value = graphName;
                    optionElement.textContent = graphName;
                    dropdown.append(optionElement);

                    // Set the new graph as the selected value
                    dropdown.value = graphName;
                    gdeState.graph = graphName;

                    // Dispatch a custom event to notify other parts of the application
                    const graphChangeEvent = new CustomEvent('graphChangeEvent', {
                        detail: {
                            graph: graphName
                        }
                    });
                    document.dispatchEvent(graphChangeEvent);
                }
                dropdown.disabled = false;
            };
        };
        const handleGraphSelect = (event, dropdown) => {
            // change gdeState.graph when it is changed
            const selectedGraph = event.target.value;
            gdeState.graph = selectedGraph;

        };


        //setupDropdown(menu, api, "Select Branch", "/gde/git/users", "branch-name-field", null, null, handleBranchCreateNew);
        //const branchDropdown = document.getElementById("branch-name-field");

        // branchDropdown.addEventListener("change", (event) => {
        //     const selectedBranch = event.target.value;
        //     if (selectedBranch && selectedBranch !== "Create New") {
        //         // Repopulate the file/graph dropdown based on the selected branch
        //         setupDropdown(menu, api, "Select Graph", "/gde/git/user_graphs", "graph-name-field", selectedBranch, null, handleGraphCreateNew);
        //     }
        //
        // });


        let tmp = setupDropdown(menu,
            api,
            "Select Graph",
            "/gde/git/user_graphs",
            "graph-name-field",
            gdeState.user,
            handleGraphCreateNew,
            handleGraphSelect);
        let graphDropdown = SetupGraphDropDown(tmp,
            null,
            menu,
            handleGraphCreateNew,
            handleGraphSelect);

    }
});
