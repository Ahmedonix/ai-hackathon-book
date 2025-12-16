import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="hero-inner">
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className="button button--secondary button--lg"
              to="/docs/intro">
              Start Learning Now
            </Link>
            <Link
              className="button button--outline button--lg"
              to="#quick-overview">
              Explore Modules
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function QuickOverviewSection() {
  return (
    <section id="quick-overview" className={styles.quickOverview}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2>Transform Your Understanding of Humanoid Robotics</h2>
            <p>
              Dive deep into the convergence of AI and robotics through a comprehensive curriculum designed to take you from ROS 2 fundamentals to advanced AI-integrated humanoid systems.
            </p>
          </div>
        </div>
        
        <div className="row margin-top--lg">
          <div className="col col--4">
            <h3>Module 1: Robotic Nervous System (ROS 2)</h3>
            <p>Master ROS 2 architecture, communication patterns, and implementation techniques to build the foundation of your intelligent robot.</p>
            <Link to="/docs/module1-ros2/index">Learn More →</Link>
          </div>
          
          <div className="col col--4">
            <h3>Module 2: Digital Twin Simulation</h3>
            <p>Create, simulate, and test humanoid robots in physics-based environments using Gazebo and Unity Robotics Hub.</p>
            <Link to="/docs/module2-simulation/index">Learn More →</Link>
          </div>
          
          <div className="col col--4">
            <h3>Module 3: AI-Robot Brain</h3>
            <p>Build an intelligent AI brain for humanoid robots using NVIDIA's Isaac platform and perception pipelines.</p>
            <Link to="/docs/module3-ai/index">Learn More →</Link>
          </div>
        </div>
        
        <div className="row margin-top--lg">
          <div className="col col--4 col--offset-2">
            <h3>Module 4: Vision-Language-Action</h3>
            <p>Integrate voice, vision, language, and action for unified cognitive robotics systems with advanced AI models.</p>
            <Link to="/docs/module4-vla/index">Learn More →</Link>
          </div>
          
          <div className="col col--4">
            <h3>Capstone Project</h3>
            <p>Synthesis your learnings in a comprehensive project integrating all modules for a complete humanoid robot system.</p>
            <Link to="/docs/capstone-project">Learn More →</Link>
          </div>
        </div>
      </div>
    </section>
  );
}

function StatsSection() {
  return (
    <section className={styles.stats}>
      <div className="container padding-vert--md text--center">
        <div className="row">
          <div className="col col--3 col--offset-1">
            <h2 className={styles.statNumber}>4</h2>
            <p className={styles.statLabel}>Comprehensive Modules</p>
          </div>
          <div className="col col--3">
            <h2 className={styles.statNumber}>20+</h2>
            <p className={styles.statLabel}>Hands-on Exercises</p>
          </div>
          <div className="col col--3">
            <h2 className={styles.statNumber}>100%</h2>
            <p className={styles.statLabel}>Open Source Curriculum</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function CallToActionSection() {
  return (
    <section className={styles.cta}>
      <div className="container text--center padding-vert--xl">
        <h2>Ready to Start Building Intelligent Humanoid Robots?</h2>
        <p>Join thousands of learners exploring the future of robotics and AI integration.</p>
        <Link
          className="button button--primary button--lg"
          to="/docs/intro">
          Begin Your Journey
        </Link>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Home - ${siteConfig.title}`}
      description="A comprehensive educational resource for humanoid robotics with ROS 2, simulation, AI perception, and VLA integration">
      <HomepageHeader />
      <main>
        <StatsSection />
        <QuickOverviewSection />
        <CallToActionSection />
      </main>
    </Layout>
  );
}