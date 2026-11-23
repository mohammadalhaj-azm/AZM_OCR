# Integration Recipe: ASP.NET Core backend + React / Angular frontend

This guide shows how a system with an **ASP.NET Core backend** and a **React or Angular frontend** uses the
**document API** (`POST /v1/documents`): a user uploads a document, and the system checks whether it is an identity
document (and which kind).

Read [INTEGRATION.md](INTEGRATION.md) first for the full API contract. This file is the ready-to-use code for that
stack. It targets **.NET 8+**.

---

## 1. Architecture: who talks to whom

```
 Browser (React / Angular)            Your ASP.NET Core backend                    OCR service
 ─────────────────────────            ─────────────────────────                    ───────────
 1. POST /api/documents  (file) ───►  2. save DocumentCheck (Pending)
                                      3. POST /v1/documents (file, X-Api-Key) ───►  queue job
 ◄── 202 { id }                       ◄── 202 { jobId }  (save jobId, Processing)
                                                                                    ... OCR + ID detection (30–90 s)
                                      4a. webhook POST /api/ocr/webhook  ◄───────── job.done
                                      4b. (fallback) background poller: GET /v1/jobs/{jobId}
                                      5. GET /v1/jobs/{jobId} → save result ──────►
                                      6. DELETE /v1/jobs/{jobId} ──────────────────►
 7. GET /api/documents/{id} (poll) ─► return status + result from YOUR database
```

**Rules:**
- **The browser never calls the OCR service directly.** The API key must stay on the server. The OCR service sits on an internal network and has no CORS.
- **The check is asynchronous.** ID detection takes about 30–90 s on CPU, which is too long to hold an HTTP request open. The backend answers the upload immediately with an id, and the frontend asks for the status.
- **Store the result in your own database**, then delete it from the OCR service. OCR results expire 1 hour after the job finished (with a webhook: 1 hour after the webhook was delivered).
- **Webhook or polling:**
  - **Webhook (4a)** is faster. It needs the OCR service to reach your backend, and your host must be allow-listed on the OCR side.
  - **Polling (4b)** needs nothing extra.
  - The code below does **both**: the webhook for speed, and the poller as a safety net. You can start with the poller only.

---

## 2. Configuration

`appsettings.json`, without secrets:
```json
{
  "OcrService": {
    "BaseUrl": "https://ocr.internal.example:8000",
    "CallbackUrl": "https://yourapp.internal.example/api/ocr/webhook",
    "MinConfidence": 0.8
  }
}
```
Keep the secrets out of the file. In development use user-secrets; in production use environment variables or a key vault:
```powershell
dotnet user-secrets set "OcrService:ApiKey" "<API_KEY>"
dotnet user-secrets set "OcrService:WebhookSecret" "<WEBHOOK_SECRET>"
# production env vars: OcrService__ApiKey, OcrService__WebhookSecret
```
Leave `CallbackUrl` empty to use polling only.

```csharp
// Ocr/OcrOptions.cs
public sealed class OcrOptions
{
    public string BaseUrl { get; set; } = "";
    public string ApiKey { get; set; } = "";
    public string? CallbackUrl { get; set; }
    public string? WebhookSecret { get; set; }
    public double MinConfidence { get; set; } = 0.8;
}
```

---

## 3. The OCR client (typed `HttpClient`)

The records below cover what this recipe needs. Every response field is typed in [`openapi.json`](openapi.json), so
you can also generate the complete models (e.g. with NSwag or Kiota), including a typed `Mrz`. Either way, keep
value lists such as `DocType` and `Status` as strings, and let unknown JSON fields be ignored (the System.Text.Json
default), so additive API changes never break you ([INTEGRATION.md §14](INTEGRATION.md#14-api-versioning-and-compatibility)).

```csharp
// Ocr/OcrModels.cs
using System.Text.Json;

public sealed record OcrJobCreated(string JobId);

public sealed record OcrJob(
    string JobId,
    string Status,                 // queued | processing | done | failed
    string? ExternalId,
    string? FullText,
    string? NormalizedText,
    List<string>? Languages,
    OcrIdDetection? IdDetection,
    OcrError? Error);

public sealed record OcrIdDetection(
    bool IsId,
    string DocType,                // national_id | passport | driver_license | residence_permit | other_id | not_id
    double Confidence,
    List<string> DecidedBy,
    List<string> Reasons,
    JsonElement? Mrz);             // present only when an MRZ was found (passports / ID cards)

public sealed record OcrError(string Code, string? Message);
public sealed record OcrErrorEnvelope(OcrError Error);

public sealed class OcrApiException(int status, string code, string message) : Exception(message)
{
    public int Status { get; } = status;
    public string Code { get; } = code;   // program against this, e.g. "unsupported_type"
}
```

```csharp
// Ocr/OcrClient.cs
using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;

public sealed class OcrClient(HttpClient http, ILogger<OcrClient> log)
{
    private const int MaxAttempts = 4;

    /// <summary>Uploads a document for text + ID detection. Returns the OCR jobId.</summary>
    /// <param name="reference">Your record id (e.g. Guid "N" format). Used as externalId, Idempotency-Key and X-Request-Id.</param>
    public async Task<string> SubmitAsync(byte[] file, string fileName, string reference, string? callbackUrl,
                                          CancellationToken ct)
    {
        using var resp = await SendWithRetryAsync(() =>
        {
            var form = new MultipartFormDataContent();
            var content = new ByteArrayContent(file);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream"); // type is detected from the bytes
            form.Add(content, "file", fileName);
            form.Add(new StringContent(reference), "externalId");
            if (!string.IsNullOrEmpty(callbackUrl)) form.Add(new StringContent(callbackUrl), "callbackUrl");

            var req = new HttpRequestMessage(HttpMethod.Post, "/v1/documents") { Content = form };
            req.Headers.Add("Idempotency-Key", reference);   // a retry never creates a second job
            req.Headers.Add("X-Request-Id", reference);
            return req;
        }, ct);

        await ThrowIfErrorAsync(resp, ct);
        var created = await resp.Content.ReadFromJsonAsync<OcrJobCreated>(ct);
        return created!.JobId;
    }

    /// <summary>Gets job status/result. Returns null if the job is unknown or expired (404).</summary>
    public async Task<OcrJob?> GetJobAsync(string jobId, int waitSeconds, CancellationToken ct)
    {
        using var resp = await SendWithRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Get, $"/v1/jobs/{jobId}?wait={waitSeconds}"), ct);
        if (resp.StatusCode == HttpStatusCode.NotFound) return null;
        await ThrowIfErrorAsync(resp, ct);
        return await resp.Content.ReadFromJsonAsync<OcrJob>(ct);
    }

    public async Task DeleteJobAsync(string jobId, CancellationToken ct)
    {
        using var resp = await SendWithRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Delete, $"/v1/jobs/{jobId}"), ct);
        if (resp.StatusCode != HttpStatusCode.NotFound) await ThrowIfErrorAsync(resp, ct);
    }

    /// <summary>Retries 503 (honouring Retry-After) and network errors/timeouts. Never retries other 4xx/5xx.</summary>
    private async Task<HttpResponseMessage> SendWithRetryAsync(Func<HttpRequestMessage> makeRequest, CancellationToken ct)
    {
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                using var req = makeRequest();                 // a fresh request (and body) for every attempt
                var resp = await http.SendAsync(req, ct);
                if (resp.StatusCode != HttpStatusCode.ServiceUnavailable || attempt >= MaxAttempts) return resp;

                var delay = resp.Headers.RetryAfter?.Delta ?? TimeSpan.FromSeconds(5);
                log.LogWarning("OCR service busy (503), retrying in {Delay}s (attempt {Attempt})", delay.TotalSeconds, attempt);
                resp.Dispose();
                await Task.Delay(delay, ct);
            }
            catch (Exception e) when (e is HttpRequestException or TaskCanceledException
                                      && attempt < MaxAttempts && !ct.IsCancellationRequested)
            {
                log.LogWarning("OCR service unreachable ({Error}), retrying (attempt {Attempt})", e.GetType().Name, attempt);
                await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)), ct);
            }
        }
    }

    private static async Task ThrowIfErrorAsync(HttpResponseMessage resp, CancellationToken ct)
    {
        if (resp.IsSuccessStatusCode) return;
        OcrError? err = null;
        try { err = (await resp.Content.ReadFromJsonAsync<OcrErrorEnvelope>(ct))?.Error; } catch { /* not JSON */ }
        throw new OcrApiException((int)resp.StatusCode, err?.Code ?? "unknown", err?.Message ?? resp.ReasonPhrase ?? "");
    }
}
```

---

## 4. Your database record (EF Core)

```csharp
// Data/DocumentCheck.cs
public enum CheckStatus { Pending, Processing, Done, Failed }

public sealed class DocumentCheck
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string UserId { get; set; } = "";
    public CheckStatus Status { get; set; } = CheckStatus.Pending;
    public string? OcrJobId { get; set; }

    // result
    public bool? IsId { get; set; }
    public string? DocType { get; set; }
    public double? Confidence { get; set; }
    public bool NeedsReview { get; set; }
    public string? ErrorCode { get; set; }
    public string? ExtractedText { get; set; }   // personal data: store only if you need it, protect it

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? CompletedAt { get; set; }
}

// in AppDbContext:  public DbSet<DocumentCheck> DocumentChecks => Set<DocumentCheck>();
// index:            modelBuilder.Entity<DocumentCheck>().HasIndex(d => d.OcrJobId);
```

---

## 5. Applying a result (shared by the webhook and the poller)

```csharp
// Ocr/OcrResultProcessor.cs
using Microsoft.Extensions.Options;

public sealed class OcrResultProcessor(AppDbContext db, OcrClient ocr, IOptions<OcrOptions> options,
                                       ILogger<OcrResultProcessor> log)
{
    /// <summary>Fetches the OCR job for this check and stores the result if it is finished. Safe to call twice.</summary>
    public async Task ProcessAsync(DocumentCheck check, CancellationToken ct)
    {
        if (check.Status is CheckStatus.Done or CheckStatus.Failed || check.OcrJobId is null) return;

        var job = await ocr.GetJobAsync(check.OcrJobId, waitSeconds: 0, ct);
        if (job is null)                                   // expired or deleted on the OCR side
        {
            Finish(check, CheckStatus.Failed, "expired");
        }
        else if (job.Status == "done" && job.IdDetection is { } det)
        {
            check.IsId = det.IsId;
            check.DocType = det.DocType;
            check.Confidence = det.Confidence;
            check.NeedsReview = det.Confidence < options.Value.MinConfidence;   // your business rule
            check.ExtractedText = job.FullText;
            Finish(check, CheckStatus.Done, null);
        }
        else if (job.Status == "failed")
        {
            Finish(check, CheckStatus.Failed, job.Error?.Code ?? "internal_error");
        }
        else return;                                       // still queued/processing

        await db.SaveChangesAsync(ct);
        if (job is not null)
        {
            try { await ocr.DeleteJobAsync(job.JobId, ct); }   // we have it now; remove it from the OCR service
            catch (Exception e) { log.LogWarning("OCR delete failed: {Error}", e.GetType().Name); } // it expires in 1 h anyway
        }
        // optional: notify the user in real time, e.g. SignalR hub: Clients.User(check.UserId).SendAsync("documentChecked", check.Id)
    }

    private static void Finish(DocumentCheck c, CheckStatus s, string? error)
    {
        c.Status = s; c.ErrorCode = error; c.CompletedAt = DateTime.UtcNow;
    }
}
```

---

## 6. The upload + status API that your frontend calls

```csharp
// Controllers/DocumentsController.cs
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

[ApiController, Route("api/documents"), Authorize]
public sealed class DocumentsController(AppDbContext db, OcrClient ocr, IOptions<OcrOptions> options) : ControllerBase
{
    private static readonly string[] Allowed = [".pdf", ".docx", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"];
    private const long MaxBytes = 20 * 1024 * 1024;

    [HttpPost, RequestSizeLimit(MaxBytes + 1024 * 1024)]
    public async Task<IActionResult> Upload(IFormFile file, CancellationToken ct)
    {
        if (file.Length == 0) return BadRequest(new { error = "The file is empty." });
        if (file.Length > MaxBytes) return BadRequest(new { error = "The file is larger than 20 MB." });
        var ext = Path.GetExtension(file.FileName).ToLowerInvariant();
        if (!Allowed.Contains(ext)) return BadRequest(new { error = "Allowed types: PDF, DOCX, JPG, PNG, WEBP, TIFF." });

        var check = new DocumentCheck { UserId = User.Identity!.Name! };
        db.DocumentChecks.Add(check);
        await db.SaveChangesAsync(ct);

        using var ms = new MemoryStream();
        await file.CopyToAsync(ms, ct);
        try
        {
            // a neutral file name: the user's file name may contain personal data
            check.OcrJobId = await ocr.SubmitAsync(ms.ToArray(), "upload" + ext, check.Id.ToString("N"),
                                                   options.Value.CallbackUrl, ct);
            check.Status = CheckStatus.Processing;
        }
        catch (OcrApiException e) when (e.Status is 400 or 413 or 415 or 422)
        {
            check.Status = CheckStatus.Failed; check.ErrorCode = e.Code; check.CompletedAt = DateTime.UtcNow;
            await db.SaveChangesAsync(ct);
            return BadRequest(new { error = UserMessage(e.Code) });
        }
        await db.SaveChangesAsync(ct);
        return Accepted(new { id = check.Id, status = check.Status.ToString() });
    }

    [HttpGet("{id:guid}")]
    public async Task<IActionResult> Get(Guid id, CancellationToken ct)
    {
        var c = await db.DocumentChecks.AsNoTracking()
                        .FirstOrDefaultAsync(d => d.Id == id && d.UserId == User.Identity!.Name, ct);
        if (c is null) return NotFound();
        return Ok(new
        {
            id = c.Id, status = c.Status.ToString(), isId = c.IsId, docType = c.DocType,
            confidence = c.Confidence, needsReview = c.NeedsReview,
            error = c.ErrorCode is null ? null : UserMessage(c.ErrorCode),
        });
    }

    private static string UserMessage(string code) => code switch
    {
        "unsupported_type" => "This file type is not supported.",
        "too_large" or "too_many_pages" => "The file is too large or has too many pages.",
        "corrupt" or "empty" => "The file is damaged, empty or password-protected.",
        "expired" => "The check timed out. Please upload the document again.",
        _ => "The document could not be checked. Please try again.",
    };
}
```

---

## 7. Option A: webhook receiver (fast)

```csharp
// Controllers/OcrWebhookController.cs
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

public sealed record OcrWebhookEvent(string Event, string DeliveryId, string JobId, string Status, string? ExternalId);

[ApiController, Route("api/ocr/webhook"), AllowAnonymous]   // authenticated by the HMAC signature instead
public sealed class OcrWebhookController(AppDbContext db, OcrResultProcessor processor,
                                         IOptions<OcrOptions> options) : ControllerBase
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    [HttpPost]
    public async Task<IActionResult> Receive(CancellationToken ct)
    {
        using var ms = new MemoryStream();
        await Request.Body.CopyToAsync(ms, ct);              // the RAW bytes: the signature covers exactly these
        var body = ms.ToArray();

        if (!IsValidSignature(options.Value.WebhookSecret ?? "", Request.Headers["X-Webhook-Timestamp"].ToString(),
                              body, Request.Headers["X-Webhook-Signature"].ToString()))
            return Unauthorized();

        var evt = JsonSerializer.Deserialize<OcrWebhookEvent>(body, Json);
        if (evt is null) return BadRequest();

        // Duplicates (retries reuse deliveryId) are harmless: ProcessAsync ignores checks that are already final.
        var check = await db.DocumentChecks.FirstOrDefaultAsync(d => d.OcrJobId == evt.JobId, ct);
        if (check is not null)
        {
            try { await processor.ProcessAsync(check, ct); }  // one quick GET; the poller retries if this fails
            catch (Exception) { /* answer 200 anyway; the background poller will pick it up */ }
        }
        return Ok();                                          // answer fast (< 10 s)
    }

    public static bool IsValidSignature(string secret, string timestamp, byte[] body, string signature)
    {
        if (secret.Length == 0 || !long.TryParse(timestamp, out var ts)) return false;
        if (Math.Abs(DateTimeOffset.UtcNow.ToUnixTimeSeconds() - ts) > 300) return false;   // reject replays

        var signed = Encoding.UTF8.GetBytes(timestamp + ".").Concat(body).ToArray();
        var expected = "sha256=" + Convert.ToHexString(
            HMACSHA256.HashData(Encoding.UTF8.GetBytes(secret), signed)).ToLowerInvariant();
        return CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(expected),
                                                       Encoding.ASCII.GetBytes(signature));
    }
}
```
**Before you use the webhook:**
- Send us the host name of `CallbackUrl`, so we can allow it for your client.
- Make sure the OCR server can reach that URL over the network.

---

## 8. Option B: background poller (safety net, or the only mode)

```csharp
// Ocr/OcrResultPoller.cs
using Microsoft.EntityFrameworkCore;

public sealed class OcrResultPoller(IServiceScopeFactory scopes, ILogger<OcrResultPoller> log) : BackgroundService
{
    // With webhooks this is only a safety net (e.g. every 30 s); without webhooks use about 5 s.
    private static readonly TimeSpan Interval = TimeSpan.FromSeconds(5);

    protected override async Task ExecuteAsync(CancellationToken stop)
    {
        while (!stop.IsCancellationRequested)
        {
            try
            {
                using var scope = scopes.CreateScope();
                var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
                var processor = scope.ServiceProvider.GetRequiredService<OcrResultProcessor>();

                var pending = await db.DocumentChecks
                    .Where(d => d.Status == CheckStatus.Processing)
                    .OrderBy(d => d.CreatedAt).Take(50).ToListAsync(stop);
                foreach (var check in pending)
                    await processor.ProcessAsync(check, stop);
            }
            catch (Exception e) when (!stop.IsCancellationRequested)
            {
                log.LogWarning("OCR poll failed: {Error}", e.GetType().Name);
            }
            await Task.Delay(Interval, stop);
        }
    }
}
```

---

## 9. `Program.cs` registration

```csharp
using Microsoft.Extensions.Options;

builder.Services.Configure<OcrOptions>(builder.Configuration.GetSection("OcrService"));

builder.Services.AddHttpClient<OcrClient>((sp, http) =>
{
    var o = sp.GetRequiredService<IOptions<OcrOptions>>().Value;
    http.BaseAddress = new Uri(o.BaseUrl);
    http.DefaultRequestHeaders.Add("X-Api-Key", o.ApiKey);
    http.Timeout = TimeSpan.FromSeconds(60);
});

builder.Services.AddScoped<OcrResultProcessor>();
builder.Services.AddHostedService<OcrResultPoller>();
```

---

## 10. Frontend

The frontend talks **only to your backend** (`/api/documents`). It uploads the file, then asks for the status every
few seconds until it is `Done` or `Failed`.

### React (TypeScript)
```tsx
import { useState } from "react";

type Check = { id: string; status: "Pending" | "Processing" | "Done" | "Failed";
               isId?: boolean; docType?: string; confidence?: number; needsReview?: boolean; error?: string };

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

export function IdDocumentUpload() {
  const [check, setCheck] = useState<Check | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setCheck({ id: "", status: "Pending" });

    const form = new FormData();
    form.append("file", file);
    const up = await fetch("/api/documents", { method: "POST", body: form, credentials: "include" });
    if (!up.ok) { setError((await up.json()).error); setCheck(null); return; }
    const { id } = await up.json();

    for (let i = 0; i < 100; i++) {                  // ~5 minutes at most
      const r = await fetch(`/api/documents/${id}`, { credentials: "include" });
      const c: Check = await r.json();
      setCheck(c);
      if (c.status === "Done" || c.status === "Failed") return;
      await sleep(3000);
    }
    setError("The check is taking longer than expected. Please refresh later.");
  }

  return (
    <div>
      <input type="file" accept=".pdf,.docx,.jpg,.jpeg,.png,.webp,.tif,.tiff" onChange={onFile} />
      {check && (check.status === "Pending" || check.status === "Processing") && <p>Checking document… (up to a minute)</p>}
      {check?.status === "Done" && (
        check.isId
          ? <p>✅ Identity document detected: {check.docType} ({Math.round((check.confidence ?? 0) * 100)}%)
               {check.needsReview && " — will be reviewed manually"}</p>
          : <p>❌ This does not look like an identity document. Please upload your ID or passport.</p>
      )}
      {check?.status === "Failed" && <p>⚠️ {check.error}</p>}
      {error && <p>⚠️ {error}</p>}
    </div>
  );
}
```

### Angular (service + component)
```ts
// document-check.service.ts
import { Injectable, inject } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { Observable, switchMap, timer, takeWhile, map } from "rxjs";

export interface Check { id: string; status: "Pending" | "Processing" | "Done" | "Failed";
  isId?: boolean; docType?: string; confidence?: number; needsReview?: boolean; error?: string; }

@Injectable({ providedIn: "root" })
export class DocumentCheckService {
  private http = inject(HttpClient);

  upload(file: File): Observable<{ id: string }> {
    const form = new FormData();
    form.append("file", file);
    return this.http.post<{ id: string }>("/api/documents", form);
  }

  /** Emits the status every 3 s, and completes after Done/Failed (the final value is emitted too). */
  watch(id: string): Observable<Check> {
    return timer(0, 3000).pipe(
      switchMap(() => this.http.get<Check>(`/api/documents/${id}`)),
      takeWhile(c => c.status !== "Done" && c.status !== "Failed", true),
    );
  }

  uploadAndWatch(file: File): Observable<Check> {
    return this.upload(file).pipe(switchMap(r => this.watch(r.id)));
  }
}
```
```ts
// id-upload.component.ts
import { Component, inject } from "@angular/core";
import { DocumentCheckService, Check } from "./document-check.service";

@Component({
  selector: "app-id-upload",
  standalone: true,
  template: `
    <input type="file" accept=".pdf,.docx,.jpg,.jpeg,.png,.webp,.tif,.tiff" (change)="onFile($event)" />
    @if (check?.status === 'Pending' || check?.status === 'Processing') { <p>Checking document… (up to a minute)</p> }
    @if (check?.status === 'Done') {
      @if (check?.isId) { <p>✅ Identity document detected: {{ check?.docType }} ({{ (check?.confidence ?? 0) * 100 | number:'1.0-0' }}%)</p> }
      @else { <p>❌ This does not look like an identity document.</p> }
    }
    @if (check?.status === 'Failed') { <p>⚠️ {{ check?.error }}</p> }
    @if (error) { <p>⚠️ {{ error }}</p> }
  `,
})
export class IdUploadComponent {
  private svc = inject(DocumentCheckService);
  check: Check | null = null;
  error: string | null = null;

  onFile(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    this.error = null;
    this.check = { id: "", status: "Pending" };
    this.svc.uploadAndWatch(file).subscribe({
      next: c => (this.check = c),
      error: err => { this.error = err.error?.error ?? "Upload failed."; this.check = null; },
    });
  }
}
```
(Add `DecimalPipe` to the component's `imports` for the `number` pipe.)

> **Real-time instead of polling:** if the app already uses **SignalR**, push a `documentChecked` event from
> `OcrResultProcessor` (see the comment there) and drop the 3-second polling loop.

---

## 11. Business rule: accept, reject or review

Decide what your application does with the result. For example, when a user must upload a passport or a national ID:

| Result | Action |
|---|---|
| `isId = true`, `docType` is one you accept, `confidence ≥ 0.8` | Accept automatically |
| `isId = true` but `confidence < 0.8` (`needsReview`) | Accept provisionally, then review manually |
| `isId = true` but the wrong `docType` (e.g. `driver_license` when you need a `passport`) | Ask the user for the right document |
| `isId = false` (`docType = not_id`) | Reject: "This is not an identity document" |
| `Failed` | Show the error message and let the user upload again |

If `idDetection.mrz` is present (passports, ID cards), use a field such as `documentNumber`, `birthDate` or
`expiryDate` only when its `checks` entry is `true`. Treat names as unverified.

---

## 12. Testing locally

1. Run the OCR service locally, e.g. `http://127.0.0.1:8000`, with `API_KEYS=change-me`. Point `OcrService:BaseUrl` to it.
2. **Poller only:** leave `CallbackUrl` empty, upload from the UI, and the status turns `Done` after about 30–90 s.
3. **Webhook:**
   - Start the OCR service with `WEBHOOK_SECRET=test-secret` and `WEBHOOK_ALLOWED_HOSTS=localhost`.
   - Set the same secret in `OcrService:WebhookSecret`.
   - Set `CallbackUrl=http://localhost:<your-port>/api/ocr/webhook`.
4. **Test the webhook controller alone** from Postman with a signed fake event. See the Postman pre-request script in the main guide's webhook section, or ask the OCR team.

## 13. Checklist

- [ ] The API key is only on the server (user-secrets, env var or key vault), never in the frontend or in git
- [ ] The frontend calls only your backend; the backend calls the OCR service
- [ ] Every upload sends `Idempotency-Key` = your record id (the client above does this)
- [ ] Retries only on 503 and network errors (the client above does this)
- [ ] The result is stored in your database, then `DELETE /v1/jobs/{id}` is called
- [ ] Webhook: signature checked on the raw body, timestamp checked, fast `200`; the poller is kept as a safety net
- [ ] No personal data in `externalId` (use your record id) and no document text in logs
- [ ] A rule for low confidence and the wrong document type is decided (see §11)
