
<!--
  The main component of the Real Estate Pricing Search & Forecast application. 
  Sets up the basic structure of the application, including the header and the main ForecastChart component.
  
  Author: ''
  Date: ''
-->

<!-- --------------------- HTML Template ---------------------- -->
<template>
  <div id="app">
    <div class="header-container">
      <img alt="Vue logo" src="./assets/logo.png" class="logo">
      <h1>Real Estate Pricing Search & Forecast</h1>
    </div>
    <ForecastChart />
  </div>
</template>

<!-- --------------------- Javascript functions ---------------------- -->
<script>
import ForecastChart from './components/Forecast.vue';

export default {
  name: 'App',
  components: {
    ForecastChart
  }
}
</script>

<!-- --------------------- CSS Styles ---------------------- -->
<style>
html, body {
  height: 100%;
  margin: 0;
  background-color: beige; /* Set background color to beige */
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 20px;
  min-height: 100%; /* Ensure the app div takes at least the full height of the viewport */
}

.header-container {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 20px; /* Add some space below the header */
}

.logo {
  width: 100px; /* Adjust the width as needed */
  height: auto; /* Maintain aspect ratio */
  margin-right: 10px; /* Add some space between the logo and the title */
}
</style>