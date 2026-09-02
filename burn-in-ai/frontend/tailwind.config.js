/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0B0F19', // Aerospace dark
        surface: '#1E293B',
        primary: '#38BDF8',
        danger: '#EF4444',
        warning: '#F59E0B',
        success: '#10B981',
      }
    },
  },
  plugins: [],
}
