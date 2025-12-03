/**
 * Interactive Three.js Background Animation
 * Creates a particle network that reacts to mouse movement.
 */

class BackgroundAnimation {
    constructor() {
        this.container = document.createElement('div');
        this.container.id = 'three-js-background';
        document.body.prepend(this.container);

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });

        this.mouseX = 0;
        this.mouseY = 0;
        this.targetX = 0;
        this.targetY = 0;

        this.windowHalfX = window.innerWidth / 2;
        this.windowHalfY = window.innerHeight / 2;

        this.init();
        this.animate();
    }

    init() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        // Create particles
        const geometry = new THREE.BufferGeometry();
        const particles = 1200; // Number of particles
        const positions = new Float32Array(particles * 3);
        const scales = new Float32Array(particles);

        for (let i = 0; i < particles; i++) {
            // Spread particles in a cloud
            positions[i * 3] = (Math.random() * 2000) - 1000;
            positions[i * 3 + 1] = (Math.random() * 2000) - 1000;
            positions[i * 3 + 2] = (Math.random() * 2000) - 1000;

            scales[i] = Math.random();
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('scale', new THREE.BufferAttribute(scales, 1));

        // Material for particles
        const material = new THREE.PointsMaterial({
            color: 0x89b1f7, // Primary color from CSS
            size: 3,
            transparent: true,
            opacity: 0.6,
            sizeAttenuation: true
        });

        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);

        this.camera.position.z = 1000;

        // Event listeners
        document.addEventListener('mousemove', this.onDocumentMouseMove.bind(this), false);
        window.addEventListener('resize', this.onWindowResize.bind(this), false);
    }

    onWindowResize() {
        this.windowHalfX = window.innerWidth / 2;
        this.windowHalfY = window.innerHeight / 2;

        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    onDocumentMouseMove(event) {
        this.mouseX = event.clientX - this.windowHalfX;
        this.mouseY = event.clientY - this.windowHalfY;
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));
        this.render();
    }

    render() {
        this.targetX = this.mouseX * 0.05;
        this.targetY = this.mouseY * 0.05;

        // Rotate the particle cloud slowly based on mouse position
        this.particles.rotation.x += 0.05 * (this.targetY * 0.001 - this.particles.rotation.x);
        this.particles.rotation.y += 0.05 * (this.targetX * 0.001 - this.particles.rotation.y);

        // Add a subtle constant rotation
        this.particles.rotation.z += 0.001;

        // Wave effect on particles
        const positions = this.particles.geometry.attributes.position.array;
        const scales = this.particles.geometry.attributes.scale.array;

        let i = 0;
        let ix = 0;

        // Optional: Animate individual particles for a "breathing" effect
        // This is computationally expensive for many particles in JS, so we keep it simple with rotation
        // But let's add a small wave if we want more "wow"

        this.renderer.render(this.scene, this.camera);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new BackgroundAnimation();
});
