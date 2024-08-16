<!-- real_estate_frontend/src/components/MapView.vue -->
<!-- This component will display the map with property locations. -->
<!-- It uses the Leaflet library to create the map and markers. -->
<!-- The properties are passed in as a prop and used to create markers on the map. -->

<!-- Displays a map with markers for property locations.
Uses Leaflet.js for map rendering. -->

<template>
    <div id="map" class="map-view"></div>
  </template>
  
  <script>
  import L from 'leaflet';
  
  export default {
    props: {
      properties: {
        type: Array,
        required: true
      }
    },
    mounted() {
      this.initMap();
    },
    methods: {
      initMap() {
        const map = L.map('map').setView([37.7749, -122.4194], 13); // Default to San Francisco
  
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
  
        this.properties.forEach(property => {
          L.marker([property.latitude, property.longitude])
            .addTo(map)
            .bindPopup(`<b>${property.address}</b><br>${property.city}, ${property.state} ${property.zip_code}`);
        });
      }
    }
  };
  </script>
  
  <style scoped>
  .map-view {
    height: 400px;
    width: 100%;
  }
  </style>