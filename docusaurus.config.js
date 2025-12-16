// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics Book',
  tagline: 'A comprehensive educational resource for humanoid robotics with ROS 2, simulation, AI perception, and VLA integration',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://speckit.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub Pages, this is usually '/<project-name>/'
  baseUrl: '/ai-hackathon-book/',

  // GitHub pages deployment config.
  organizationName: '[YOUR_GITHUB_USERNAME_OR_ORG]', // Usually your GitHub org/user name.
  projectName: 'ai-hackathon-book', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/speckit/ai-hackathon-book/tree/main/',
        },
        blog: false, // Disable blog for educational book
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'AI Robotics Book',
        logo: {
          alt: 'Robotics Book Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Modules',
          },
          {
            href: 'https://github.com/speckit/ai-hackathon-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Modules',
            items: [
              {
                label: 'Module 1: ROS 2 Fundamentals',
                to: '/docs/module1-ros2',
              },
              {
                label: 'Module 2: Digital Twin Simulation',
                to: '/docs/module2-simulation',
              },
              {
                label: 'Module 3: AI-Robot Brain',
                to: '/docs/module3-ai',
              },
              {
                label: 'Module 4: Vision-Language-Action',
                to: '/docs/module4-vla',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/speckit/ai-hackathon-book',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Speckit AI Robotics. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;