<template>
  <h1>採点をする</h1>
  <div>
    <AssessmentRequest/>
  </div>
</template>

<script>
import AssessmentRequest from "@/components/AssessmentRequest.vue";
export default {
  name: "Assessment",
  components: {
    AssessmentRequest
  },
};
</script>

