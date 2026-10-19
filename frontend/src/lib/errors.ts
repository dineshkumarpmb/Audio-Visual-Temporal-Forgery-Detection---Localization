/**
 * Every API error code (api/errors.py, api/services/upload.py, inference_service.py) in
 * plain words: what happened, and what the person can do about it. A non-expert must be
 * able to act on these without knowing what a codec is (P14-12).
 */
export interface FriendlyError {
  title: string;
  advice: string;
}

const MESSAGES: Record<string, FriendlyError> = {
  file_too_large: {
    title: "This file is too large",
    advice: "Trim the video or export it at a lower quality, then try again.",
  },
  unsupported_format: {
    title: "This file type isn't supported",
    advice: "Use an MP4, MOV, AVI or MKV video.",
  },
  unsupported_media_type: {
    title: "This doesn't look like a video file",
    advice: "Choose an MP4, MOV, AVI or MKV video.",
  },
  format_mismatch: {
    title: "The file's name doesn't match its contents",
    advice: "The file extension says one format but the file is another. Re-export it as MP4.",
  },
  unsupported_codec: {
    title: "This video uses an unusual encoding",
    advice: "Re-export it as a standard MP4 (H.264) and try again.",
  },
  corrupt_video: {
    title: "The video couldn't be read",
    advice: "The file may be damaged or incomplete. Try downloading or exporting it again.",
  },
  no_video_stream: {
    title: "There is no picture in this file",
    advice: "Upload a video, not an audio-only file.",
  },
  no_audio: {
    title: "This video has no sound",
    advice: "The detector compares the face with the voice, so it needs the soundtrack.",
  },
  duration_out_of_range: {
    title: "The video is too short or too long",
    advice: "Upload a clip between half a second and five minutes long.",
  },
  no_face: {
    title: "No face was found",
    advice: "The detector checks a person speaking on camera. Use a clip where a face is clearly visible.",
  },
  timeout: {
    title: "The analysis took too long",
    advice: "Try a shorter clip.",
  },
  queue_full: {
    title: "The server is busy",
    advice: "Other videos are being analysed. Wait a few seconds and try again.",
  },
  server_unreachable: {
    title: "Can't reach the analysis server",
    advice: "Make sure the server is running (make api), then try again.",
  },
  empty_file: {
    title: "The file is empty",
    advice: "Choose the video again; the file may not have finished downloading.",
  },
  missing_file: { title: "No file was sent", advice: "Choose a video and try again." },
  malformed_upload: {
    title: "The upload was interrupted",
    advice: "Check your connection and try again.",
  },
  job_not_found: {
    title: "This result has expired",
    advice: "Results are kept for an hour. Upload the video again.",
  },
  aborted: { title: "Upload cancelled", advice: "Choose a video to start again." },
  report_unavailable: {
    title: "This server can't make PDF reports yet",
    advice:
      "The analysis server was started before the report feature existed. Stop it (Ctrl+C) and run make api again, then analyse the video again.",
  },
  job_not_ready: {
    title: "The analysis hasn't finished",
    advice: "Wait for the result to appear, then download the report.",
  },
};

const FALLBACK: FriendlyError = {
  title: "Something went wrong",
  advice: "Try again. If it keeps happening, the server log has the details.",
};

export function friendlyError(code: string): FriendlyError {
  return MESSAGES[code] ?? FALLBACK;
}
