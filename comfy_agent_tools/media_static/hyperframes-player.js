class HyperframesPlayer extends HTMLElement {
  static get observedAttributes() {
    return ["src"];
  }

  connectedCallback() {
    this.render();
  }

  attributeChangedCallback() {
    this.render();
  }

  render() {
    const src = this.getAttribute("src");
    if (!src) return;
    this.innerHTML = "";
    const frame = document.createElement("iframe");
    frame.src = src;
    frame.title = this.getAttribute("title") || "HyperFrames composition";
    frame.loading = "lazy";
    frame.sandbox = "allow-scripts allow-same-origin";
    frame.style.width = "100%";
    frame.style.height = "100%";
    frame.style.border = "0";
    frame.style.background = "#101114";
    this.appendChild(frame);
    this.frame = frame;
  }

  play() {
    this.frame?.contentWindow?.postMessage({ type: "hyperframes:play" }, "*");
  }

  pause() {
    this.frame?.contentWindow?.postMessage({ type: "hyperframes:pause" }, "*");
  }

  seek(time) {
    this.frame?.contentWindow?.postMessage({ type: "hyperframes:seek", time }, "*");
  }
}

customElements.define("hyperframes-player", HyperframesPlayer);
