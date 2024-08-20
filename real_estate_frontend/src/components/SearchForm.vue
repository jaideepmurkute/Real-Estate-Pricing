<!-- real_estate_frontend/src/components/SearchForm.vue -->
<!-- This component will handle the search form where users can input their search criteria. -->

<!-- Allows users to input search criteria (city, state, zip code).
Emits a search event with the search criteria. -->

<template>
  <div class="search-form">
    <h2>Search Properties</h2>
    <form @submit.prevent="searchProperties">
      <div>
        <label for="city">City:</label>
        <input type="text" id="city" v-model="city" />
      </div>
      <div>
        <label for="state">State:</label>
        <input type="text" id="state" v-model="state" />
      </div>
      <div>
        <label for="zip_code">Zip Code:</label>
        <input type="text" id="zip_code" v-model="zip_code" />
      </div>
      <button type="submit">Search</button>
    </form>
  </div>
</template>

<script>
export default {
  data() {
    return {
      city: '',
      state: '',
      zip_code: ''
    };
  },
  methods: {
    searchProperties() {
      // Emit search event with search criteria
      this.$emit('search', {
        city: this.city,
        state: this.state,
        zip_code: this.zip_code
      });
    }
  }
};
</script>

<style scoped>
.search-form {
  margin: 20px;
}
</style>