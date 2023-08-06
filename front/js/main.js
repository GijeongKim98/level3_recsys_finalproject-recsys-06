function login() {
    const input = document.getElementById("login_input").value;
    localStorage.setItem('user', input);
    window.location.href = "home.html"
}