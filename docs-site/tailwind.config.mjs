import starlightPlugin from '@astrojs/starlight-tailwind';

// Pydantic-inspired color palette
const accent = {
  200: '#f9a8d4', // pink-300
  600: '#db2777', // pink-600
  900: '#831843', // pink-900
  950: '#500724', // pink-950
};

const gray = {
  100: '#f5f5f5',
  200: '#e5e5e5',
  300: '#d4d4d4',
  400: '#a3a3a3',
  500: '#737373',
  700: '#404040',
  800: '#262626',
  900: '#171717',
};

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        accent,
        gray,
      },
    },
  },
  plugins: [starlightPlugin()],
};
