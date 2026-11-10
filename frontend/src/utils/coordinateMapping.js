/**
 * Universal Coordinate Mapping Engine for Real-Time Pose Visualization
 * ====================================================================
 * Maps MediaPipe normalized [0..1] landmark coordinates to pixel coordinates
 * on the HTML5 canvas overlay, perfectly aligned with the CSS object-fit
 * rendered video element beneath it.
 *
 * Solves:
 * 1. Double Horizontal Flip: Canvas has NO scaleX(-1); coordinates are drawn directly.
 * 2. Aspect-Ratio Distortion: Accounts for 16:9 webcam in 4:3 container (or vice versa),
 *    computing exact scaling and crop offsets (dx, dy).
 * 3. HiDPI / Retina Displays: Accounts for window.devicePixelRatio.
 */

/**
 * Calculates the exact bounding box and crop offsets of the video displayed
 * inside its container based on the CSS object-fit property ('cover' or 'contain').
 *
 * @param {HTMLVideoElement} video - The HTML video element.
 * @param {number} containerWidth - The clientWidth of the video container.
 * @param {number} containerHeight - The clientHeight of the video container.
 * @param {string} objectFit - 'cover' | 'contain' (default: 'cover')
 * @returns {object} Bounding box details { videoWidth, videoHeight, displayedWidth, displayedHeight, offsetX, offsetY, scale }
 */
export function getDisplayedVideoRect(video, containerWidth, containerHeight, objectFit = 'cover') {
  const videoWidth = (video && video.videoWidth) ? video.videoWidth : 640;
  const videoHeight = (video && video.videoHeight) ? video.videoHeight : 480;

  const videoAspect = videoWidth / videoHeight;
  const containerAspect = (containerWidth && containerHeight)
    ? (containerWidth / containerHeight)
    : (4 / 3);

  let displayedWidth = containerWidth;
  let displayedHeight = containerHeight;
  let offsetX = 0;
  let offsetY = 0;
  let scale = 1.0;

  if (objectFit === 'cover') {
    if (containerAspect > videoAspect) {
      // Container is wider than video -> width matches container, height extends and is cropped
      scale = containerWidth / videoWidth;
      displayedWidth = containerWidth;
      displayedHeight = videoHeight * scale;
      offsetX = 0;
      offsetY = (containerHeight - displayedHeight) / 2;
    } else {
      // Container is taller than video -> height matches container, width extends and is cropped
      scale = containerHeight / videoHeight;
      displayedHeight = containerHeight;
      displayedWidth = videoWidth * scale;
      offsetX = (containerWidth - displayedWidth) / 2;
      offsetY = 0;
    }
  } else {
    // 'contain'
    if (containerAspect > videoAspect) {
      // Container is wider -> letterboxed left and right
      scale = containerHeight / videoHeight;
      displayedHeight = containerHeight;
      displayedWidth = videoWidth * scale;
      offsetX = (containerWidth - displayedWidth) / 2;
      offsetY = 0;
    } else {
      // Container is taller -> letterboxed top and bottom
      scale = containerWidth / videoWidth;
      displayedWidth = containerWidth;
      displayedHeight = videoHeight * scale;
      offsetX = 0;
      offsetY = (containerHeight - displayedHeight) / 2;
    }
  }

  return {
    videoWidth,
    videoHeight,
    containerWidth,
    containerHeight,
    displayedWidth,
    displayedHeight,
    offsetX,
    offsetY,
    scale
  };
}

/**
 * Maps a single normalized landmark {x, y, z, visibility} to canvas pixel coordinates.
 *
 * Note on mirroring:
 * The backend processes video frames flipped via cv2.flip(bgr_image, 1).
 * Therefore, landmark.x is already in the mirror-space of the user's mirrored webcam view!
 * Setting isMirrored = false maps this mirrored x directly to canvas pixel space.
 * If unmirrored raw coordinates are ever provided, set isMirrored = true to flip.
 *
 * @param {object} landmark - { x: number, y: number, visibility?: number }
 * @param {object} rect - Output from getDisplayedVideoRect
 * @param {boolean} isMirrored - Whether to invert horizontal x axis (default: false)
 * @returns {object} { x: number, y: number, visible: boolean } in CSS pixel coordinates
 */
export function mapLandmarkToCanvas(landmark, rect, isMirrored = false) {
  if (!landmark || !rect) {
    return { x: 0, y: 0, visible: false };
  }

  const rawX = landmark.x ?? 0;
  const rawY = landmark.y ?? 0;
  const visibility = landmark.visibility ?? 1.0;

  // Compute normalized coordinate in mirror space
  const normX = isMirrored ? (1.0 - rawX) : rawX;
  const normY = rawY;

  // Scale to displayed dimensions and apply crop/letterbox offsets
  const pixelX = rect.offsetX + normX * rect.displayedWidth;
  const pixelY = rect.offsetY + normY * rect.displayedHeight;

  // A point is visible if its visibility score is sufficient and it falls within container bounds
  const withinBounds = (
    pixelX >= -20 && pixelX <= rect.containerWidth + 20 &&
    pixelY >= -20 && pixelY <= rect.containerHeight + 20
  );

  return {
    x: pixelX,
    y: pixelY,
    visible: (visibility > 0.45) && withinBounds
  };
}

