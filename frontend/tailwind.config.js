/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        agentic: {
          dark: '#ffffffff',
          panel: '#ffffffff',
          blue: '#1e3a8a',
          cyan: '#00FFFF',
          emerald: '#00FFBF',
        }
      },
      backgroundImage: {
        'gradient-agentic': 'linear-gradient(to right, #1e3a8a, #00FFFF, #00FFBF)',
        'gradient-card': 'linear-gradient(135deg, rgba(30, 58, 138, 0.4), rgba(0, 255, 255, 0.1), rgba(0, 255, 191, 0.05))',
      },
      boxShadow: {
        'glow-cyan': '0 0 15px rgba(0, 255, 255, 0.4)',
        'glow-emerald': '0 0 15px rgba(0, 255, 191, 0.4)',
        'glow-subtle': 'inset 0 0 20px rgba(0, 255, 255, 0.05)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
