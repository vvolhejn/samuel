/** Browser mic processing, toggleable so we can A/B it against training audio,
 * which has none of it. vad-web hardcodes all three on; we pass our own
 * getStream/resumeStream instead (see startMic). */
export type MicProcessing = {
  echoCancellation: boolean;
  autoGainControl: boolean;
  noiseSuppression: boolean;
};

export const MIC_PROCESSING_DEFAULTS: MicProcessing = {
  echoCancellation: true,
  autoGainControl: true,
  noiseSuppression: true,
};

export const MIC_PROCESSING_LABELS: Array<{
  key: keyof MicProcessing;
  label: string;
}> = [
  { key: "echoCancellation", label: "echo cancellation" },
  { key: "autoGainControl", label: "auto gain control" },
  { key: "noiseSuppression", label: "noise suppression" },
];
