package io.github.jafarsalim.phishguard;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Minimal Java client for the phishguard HTTP API (no dependencies, Java 11+).
 *
 * <pre>
 *   javac -d out java/src/main/java/io/github/jafarsalim/phishguard/PhishGuardClient.java
 *   java -cp out io.github.jafarsalim.phishguard.PhishGuardClient http://localhost:8000 message.eml [...]
 * </pre>
 *
 * Prints one line per message: verdict, score and the three strongest reasons.
 * Use {@link #scan(Path)} from your own code (a mail gateway, a Spring service, ...).
 */
public final class PhishGuardClient {
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    private final URI scanUri;

    public PhishGuardClient(String baseUrl) {
        this.scanUri = URI.create(baseUrl.replaceAll("/+$", "") + "/scan");
    }

    /** Result of one scan: the raw JSON plus the fields most callers need. */
    public record Result(String verdict, double score, String json) {}

    public Result scan(Path eml) throws IOException, InterruptedException {
        HttpRequest req = HttpRequest.newBuilder(scanUri).timeout(Duration.ofSeconds(30))
                .header("Content-Type", "message/rfc822")
                .POST(HttpRequest.BodyPublishers.ofByteArray(Files.readAllBytes(eml))).build();
        HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (res.statusCode() != 200) {
            throw new IOException("phishguard returned HTTP " + res.statusCode() + ": " + res.body());
        }
        String body = res.body();
        return new Result(field(body, "\"verdict\"\\s*:\\s*\"([^\"]+)\""),
                Double.parseDouble(field(body, "\"score\"\\s*:\\s*([0-9.eE+-]+)")), body);
    }

    private static String field(String json, String regex) {
        Matcher m = Pattern.compile(regex).matcher(json);
        return m.find() ? m.group(1) : "";
    }

    static String topReasons(String json, int k) {
        Matcher m = Pattern.compile("\"description\"\\s*:\\s*\"([^\"]+)\"").matcher(json);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < k && m.find(); i++) {
            sb.append(i == 0 ? "" : "; ").append(m.group(1));
        }
        return sb.toString();
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: PhishGuardClient <api-base-url> <message.eml> [more.eml ...]");
            System.exit(2);
        }
        PhishGuardClient c = new PhishGuardClient(args[0]);
        for (int i = 1; i < args.length; i++) {
            Result r = c.scan(Path.of(args[i]));
            System.out.printf("%-11s %.3f  %s  | %s%n", r.verdict(), r.score(), Path.of(args[i]).getFileName(), topReasons(r.json(), 3));
        }
    }
}
