const form = document.getElementById("form");
const err = document.getElementById("err");
try {
  const theme = localStorage.getItem("ocr-admin-theme");
  if (theme) document.documentElement.dataset.theme = theme;
} catch { /* storage unavailable */ }

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  err.textContent = "";
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    const r = await fetch("/admin/api/login", {
      method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-Admin-Request": "1" },
      body: JSON.stringify({ username: form.username.value.trim(), password: form.password.value }),
    });
    if (r.ok) {
      location.href = "/admin";
      return;
    }
    const body = await r.json().catch(() => ({}));
    err.textContent = r.status === 429
      ? `Too many failed attempts. Try again in ${r.headers.get("Retry-After") || "a few"} seconds.`
      : (body.error && body.error.message) || "Sign-in failed.";
  } catch {
    err.textContent = "The server can't be reached.";
  } finally {
    button.disabled = false;
  }
});
