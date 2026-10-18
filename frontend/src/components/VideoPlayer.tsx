import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

export interface PlayerHandle {
  seek: (time: number, play?: boolean) => void;
}

interface Props {
  src: string;
  onTime: (time: number) => void;
}

/**
 * The uploaded file, played locally from an object URL -- the server never sends the video
 * back (P14-5). `seek` is how the timeline, the chart and the segment list drive it.
 */
export const VideoPlayer = forwardRef<PlayerHandle, Props>(function VideoPlayer({ src, onTime }, ref) {
  const video = useRef<HTMLVideoElement>(null);

  useImperativeHandle(ref, () => ({
    seek(time, play = true) {
      const v = video.current;
      if (!v) return;
      v.currentTime = time;
      onTime(time);
      // Autoplay may be refused (and older engines return no promise); seeking still works.
      if (play) v.play()?.catch(() => undefined);
    },
  }));

  useEffect(() => {
    const v = video.current;
    if (!v) return;
    // timeupdate fires ~4x/s; rAF while playing keeps the playhead smooth.
    let raf = 0;
    const loop = () => {
      onTime(v.currentTime);
      if (!v.paused) raf = requestAnimationFrame(loop);
    };
    const start = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(loop);
    };
    const once = () => onTime(v.currentTime);
    v.addEventListener("play", start);
    v.addEventListener("seeked", once);
    v.addEventListener("pause", once);
    return () => {
      cancelAnimationFrame(raf);
      v.removeEventListener("play", start);
      v.removeEventListener("seeked", once);
      v.removeEventListener("pause", once);
    };
  }, [onTime]);

  return (
    <video
      ref={video}
      src={src}
      controls
      playsInline
      preload="metadata"
      className="aspect-video w-full rounded-lg bg-black"
      data-testid="video"
    />
  );
});
