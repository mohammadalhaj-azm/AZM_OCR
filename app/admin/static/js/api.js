// JSON API client for /admin/api. Mutating requests carry the CSRF header; a 401 goes back to the login page.

export class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function call(method, path, body) {
  const headers = { Accept: "application/json" };
  if (method !== "GET") headers["X-Admin-Request"] = "1";
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let r;
  try {
    r = await fetch(`/admin/api${path}`, {
      method, headers, credentials: "same-origin", body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "network", "The server can't be reached.");
  }
  if (r.status === 401 && path !== "/login") {
    location.href = "/admin/login";
    throw new ApiError(401, "unauthorized", "Session expired.");
  }
  if (r.status === 204) return null;
  const type = r.headers.get("Content-Type") || "";
  const data = type.includes("json") ? await r.json().catch(() => null) : await r.text();
  if (!r.ok) {
    const e = (data && data.error) || {};
    throw new ApiError(r.status, e.code || "error", e.message || `Request failed (${r.status}).`);
  }
  return data;
}

export function qs(params) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null && v !== "") p.set(k, v);
  }
  const s = p.toString();
  return s ? `?${s}` : "";
}

export const api = {
  get: (path, params) => call("GET", path + qs(params)),
  post: (path, body) => call("POST", path, body ?? {}),
  put: (path, body) => call("PUT", path, body),
  del: (path) => call("DELETE", path),
};
