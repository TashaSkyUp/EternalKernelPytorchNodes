import {app} from "../../../scripts/app.js";
import {api} from "../../../scripts/api.js";
import {setupButtons, setupModal, setupDropdown} from './ui_helpers.js';

// create a place to store some variables, including security_token
const gde = {
    security_token: null,
}
gde.login_div = document.createElement("div");
gde.login_div.id = "gde-login";

gde.usernameField = document.createElement("input");
gde.usernameField.id = "username-field";
gde.usernameField.placeholder = "Username";
//add a logout button
gde.logoutButton = document.createElement("button");
gde.logoutButton.id = "logout-button";
gde.logoutButton.innerText = "Logout";
gde.logoutButton.addEventListener("click", async () => {
        //clear the token
        gde.security_token = null;
        //remove the logout button
        gde.logoutButton.remove();
        //add the login button and fields
        gde.login_div.append(gde.usernameField);
        gde.login_div.append(gde.passwordField);
        gde.login_div.append(gde.loginButton);
        // clear the graph dropdown
        const graphDropdown = document.getElementById("graph-name-field");
        if (graphDropdown) {
            // remove all options without removing the first two
            while (graphDropdown.options.length > 2) {
                graphDropdown.remove(2);
            }
        }
        // clear the branch dropdown
        const branchDropdown = document.getElementById("branch-name-field");
        if (branchDropdown) {
            // remove all options without removing the first two
            while (branchDropdown.options.length > 2) {
                branchDropdown.remove(2);
            }
        }

        // tell the server we logged out
        await api.fetchApi("/gde/logout", {
                method: "POST",
                body: JSON.stringify({
                    client_id: api.clientId,
                })
            }
        );
    }
);
gde.passwordField = document.createElement("input");
gde.passwordField.id = "password-field";
gde.passwordField.placeholder = "Password";
gde.passwordField.type = "password";
gde.loginButton = document.createElement("button");
gde.loginButton.id = "login-button";
gde.loginButton.style.color = "var(--input-text)";
gde.loginButton.style.backgroundColor = "var(--comfy-input-bg)";
gde.loginButton.style.borderRadius = "8px";
gde.loginButton.style.borderColor = "var(--border-color)";
gde.loginButton.style.borderStyle = "solid";
gde.loginButton.style.marginTop = "2px";
gde.loginButton.style.width = "100%";
gde.loginButton.innerText = "Login";
gde.loginButton.addEventListener("click", async () => {
        const username = document.getElementById("username-field").value;
        const password = document.getElementById("password-field").value;
        const res = await api.fetchApi("/gde/login", {
                    method: "POST",
                    body: JSON.stringify({
                            username: username,
                            password: password,
                            client_id: api.clientId,
                        }
                    )
                }
            )
        ;
        if (res.status !== 200) {
            alert("Login failed");
        } else {
            //read the json response, token will be under security
            const json = await res.json();

            //set the token it is somewhere in the json under security
            gde.security_token = json.security;

            //now that we are logged in remove the login button and fields
            gde.usernameField.remove();
            gde.passwordField.remove();
            gde.loginButton.remove();


            gde.login_div.append(gde.logoutButton);


        }
    }
);

gde.get_server_state = async (client_id) => {
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

// create a dummy function
gde.setup_login_form = async () => {
    const menu = document.querySelector(".comfy-menu");
    // create a div in the menu for this extension


    //need to check with server for the status of our current api.clientId
    const cid = api.clientId || api.socket.url.split("=")[1];
    const server_state = await gde.get_server_state(cid).then((server_state) => {
        const logged_in = server_state.logged_in;

        const separator = document.createElement("hr");
        separator.style.margin = "20px 0";
        separator.style.width = "100%";
        menu.append(separator);
        menu.append(gde.login_div);

        if (logged_in) {
            gde.login_div.append(gde.logoutButton);

        } else {
            gde.login_div.append(gde.usernameField);
            gde.login_div.append(gde.passwordField);
            gde.login_div.append(gde.loginButton);

        }
    });

}
gde.hijack_queueprompt = async () => {
    // is this actually needed?
}
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
            await gde.setup_login_form();
            await gde.hijack_queueprompt();
        }, 10);
    }


});

app.registerExtension({
    name: "GDE.UI.Git",
    async setup() {
        setupModal();
        const separator = document.createElement("hr");
        const menu = document.querySelector(".comfy-menu");
        separator.style.margin = "20px 0";
        separator.style.width = "100%";
        menu.append(separator);
        setupButtons(menu, app, api);
        const handleBranchCreateNew = (event, dropdown) => {
            // Custom logic for when "Create New" is selected in the branch dropdown
            showModal('branch-name-field'); // Show the modal for creating a new branch
            dropdown.value = ""; // Reset the dropdown value
            // Clear the file dropdown
            const fileDropdown = document.getElementById("graph-name-field");
            if (fileDropdown) {
                // remove all options without removing the first two
                while (fileDropdown.options.length > 2) {
                    fileDropdown.remove(2);
                }
            }
        };

        const handleGraphCreateNew = (event, dropdown) => {
            // Custom logic for when "Create New" is selected in the branch dropdown
            showModal('graph-name-field'); // Show the modal for creating a new branch
            dropdown.value = ""; // Reset the dropdown value
            // Clear the file dropdown
            const fileDropdown = document.getElementById("graph-name-field");
        };

        setupDropdown(menu, api, "Select Branch", "/gde/git/users", "branch-name-field", null, null, handleBranchCreateNew);
        const branchDropdown = document.getElementById("branch-name-field");

        branchDropdown.addEventListener("change", (event) => {
            const selectedBranch = event.target.value;
            if (selectedBranch && selectedBranch !== "Create New") {
                // Repopulate the file/graph dropdown based on the selected branch
                setupDropdown(menu, api, "Select Graph", "/gde/git/user_graphs", "graph-name-field", selectedBranch, null, handleGraphCreateNew);
            }

        });


        setupDropdown(menu, api, "Select Graph", "/gde/git/user_graphs", "graph-name-field", null, null, handleGraphCreateNew);

    }
});
